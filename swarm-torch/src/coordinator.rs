//! Unified Swarm Round Coordinator (`ADR-0023` §1, `P2-07`..`P2-09`, `P2-12`, `W10-01`).
//!
//! Bridges the cryptographic admission gate (`ReplayProtection` + `CohortAllowlist`),
//! `UpdateTransform` audit pipeline, `RobustAggregator` / `WeightingPolicyV1` math,
//! `SwarmModel` / `SwarmOptimizer` / `FitnessFunction` execution, and `DataOpsSession` +
//! physical CAS checkpoint persistence (`blobs/sha256/<digest>`) into a single deterministic,
//! fail-closed round state machine.

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use sha2::{Digest, Sha256};
use swarm_torch_core::aggregation::{
    aggregate_with_bucketing_v1, apply_update_transforms, bucket_updates_v1, AggregationBudget,
    BucketExecutionReport, BucketRemainderPolicy, BucketingConfig, CenteredClipping,
    CoordinateMedian, FedAvg, PostBucketingRule, RobustAggregation, RobustAggregator, TrimmedMean,
};
#[cfg(feature = "robust-aggregation")]
use swarm_torch_core::aggregation::{Bulyan, BulyanProfile, Krum};
use swarm_torch_core::compression::CompressedGradient;
use swarm_torch_core::crypto::{sha256_slices, KeyPair, SessionHandshakeChallengeV2, Signature};
use swarm_torch_core::dataops::{
    AuthModeMarker, CacheDecisionV0, SourceDescriptorV0, TransformAuditV0, TrustClass,
};
use swarm_torch_core::observe::{
    AttrMap, AttrValue, EventRecord, MetricRecord, RunEventEmitter, RunId, SpanId, SpanRecord,
    TraceId,
};
use swarm_torch_core::replay::{
    CohortAllowlist, PreparedReplayAdmission, ReplayError, ReplayProtection, ReplayRecordKeyV2,
};
use swarm_torch_core::run_graph::{
    node_def_hash_v1, node_id_from_key, AssetRefV1, CanonParams, ExecutionTrust, NodeV1, OpKind,
};
use swarm_torch_core::traits::{
    FitnessFunction, GradientUpdate, OptimizerStateSnapshotV1, PeerId, SwarmModel, SwarmOptimizer,
    UpdateTransform,
};
use swarm_torch_core::Error as CoreError;
use swarm_torch_models::lora::LoraAdapterPairV1;
use swarm_torch_net::gossip::{
    BundledGossipDeliveryV2, GossipVerifyError, InnerUpdateV2, PayloadKindV1,
};

use crate::artifacts::{
    get_verified_cclip_center_v1, get_verified_checkpoint_model,
    get_verified_historical_anchors_v1, get_verified_optimizer_state_snapshot_v1,
    pre_release_debit_dp_accountant, put_cclip_center_v1, put_checkpoint_model,
    put_coordinator_resume_state_v2, put_historical_anchors_v1, put_optimizer_state_snapshot_v1,
    reconstruct_dp_accountant_from_ledger, write_committed_head_v1, write_round_resume_state,
    CheckpointContentDigestV1, CheckpointRecipeKeyV1, CommittedHeadV1, CoordinatorResumeStateV2,
    CryptoErrorOrIo, DataOpsSession, OutputSpec, RestoreModeV1, RoundResumeStateV1, TrustPolicy,
    COMMITTED_HEAD_SCHEMA_V1, COORDINATOR_RESUME_STATE_SCHEMA_V2, ROUND_RESUME_STATE_SCHEMA_V1,
};

/// Domain tag for canonical peer round submission digests (`ADR-0023` §1.3).
pub const PEER_SUBMISSION_DIGEST_DOMAIN_V1: &[u8] = b"swarmtorch.round.submission.v1";

/// Typed update algebra for federated/swarm rounds (`ADR-0023` §1.1, `W10-01`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UpdateKindV1 {
    /// Multi-step local model delta $\Delta_i = w_i^{\text{local}} - w_{\text{parent}}$,
    /// applied on the coordinator as $w_{\text{candidate}} = w_{\text{parent}} + \eta_{\text{server}} a$
    /// in widened `f64` precision.
    ///
    /// Local `FedProx` proximal regularization ($\frac{\mu}{2}\|w - w_{\text{parent}}\|^2$)
    /// is applied inside the participant's local objective and never double-penalized at the server.
    ModelDeltaV1 {
        /// Server step size $\eta_{\text{server}}$ (default `1.0`).
        server_learning_rate: f32,
    },
    /// Single-step gradient vector $g_i$, applied once on the coordinator via
    /// [`SwarmOptimizer::step`].
    GradientV1,
}

impl Default for UpdateKindV1 {
    fn default() -> Self {
        Self::ModelDeltaV1 {
            server_learning_rate: 1.0,
        }
    }
}

/// Trust-bounded peer weighting policy (`ADR-0023` §1.2, `W10-01`).
///
/// Prevents Byzantine sample-weight spoofing (`reported_examples` inflation) by never
/// trusting wire-reported sample counts in aggregation math.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, Default)]
pub enum WeightingPolicyV1 {
    /// Uniform 1-vote-per-admitted-peer weighting ($\alpha_i = 1 / |S^{(r)}|$).
    ///
    /// Mandatory for all Byzantine-resilient aggregators (`TrimmedMean`, `Median`, `Krum`,
    /// `MultiKrum`, `Bulyan`) and the default for `FedAvg`.
    #[default]
    EqualPeerVote,
    /// Opt-in coordinator-certified dataset weighting for `FedAvg` only.
    ///
    /// Weights are taken strictly from coordinator-trusted registry metadata (`peer_weights`),
    /// capped per peer at `max_weight_share` in `(0.0, 1.0]`, and normalized in `f64` over the
    /// admitted cohort.
    RegistryCertified {
        /// Coordinator-certified dataset weight ($N_k^{\text{reg}} > 0$) per authorized peer.
        peer_weights: BTreeMap<PeerId, u64>,
        /// Upper bound on any single peer's pre-renormalization weight share in `(0.0, 1.0]`.
        max_weight_share: f64,
    },
}

/// Configuration for [`SwarmRoundCoordinator`] (`ADR-0023` §1).
#[derive(Debug, Clone)]
pub struct RoundCoordinatorConfig {
    /// Update algebra (`ModelDeltaV1` default vs. `GradientV1`).
    pub update_kind: UpdateKindV1,
    /// Robust aggregation strategy.
    pub aggregation: RobustAggregation,
    /// Peer weighting policy (`EqualPeerVote` default vs. `RegistryCertified` for `FedAvg`).
    pub weighting_policy: WeightingPolicyV1,
    /// Authorized cohort allowlist.
    pub authorized_cohort: BTreeSet<PeerId>,
    /// Minimum number of non-equivocating admitted peers required to commit a round.
    pub min_quorum: usize,
    /// Maximum allowed round staleness ($r_{\text{current}} - r_{\text{msg}} \le \tau_{\max}$).
    /// Defaults to `0` (strict synchronous round matching).
    pub max_staleness: u64,
    /// Maximum clock skew in seconds for `ReplayProtection`.
    pub max_clock_skew_secs: u32,
}

/// Configuration validation errors for [`RoundCoordinatorConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum RoundConfigError {
    EmptyCohort,
    InvalidQuorum {
        min_quorum: usize,
        cohort_len: usize,
    },
    InvalidServerLearningRate(f32),
    IncompatibleWeightingPolicy,
    InvalidMaxWeightShare(f64),
    MissingOrZeroCertifiedWeight(PeerId),
    /// Caller requested certified Differential Privacy enforcement before a noise-injecting mechanism (`DpFedAvgV1`) is active (`ST-02`).
    UnsupportedCertifiedDifferentialPrivacy,
}

impl core::fmt::Display for RoundConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyCohort => write!(f, "authorized_cohort must not be empty"),
            Self::InvalidQuorum {
                min_quorum,
                cohort_len,
            } => write!(
                f,
                "min_quorum ({min_quorum}) must be in 1..={cohort_len}"
            ),
            Self::InvalidServerLearningRate(lr) => {
                write!(f, "server_learning_rate must be finite and > 0.0, got {lr}")
            }
            Self::IncompatibleWeightingPolicy => write!(
                f,
                "WeightingPolicyV1::RegistryCertified is only permitted with RobustAggregation::FedAvg"
            ),
            Self::InvalidMaxWeightShare(s) => {
                write!(f, "max_weight_share must be finite and in (0.0, 1.0], got {s}")
            }
            Self::MissingOrZeroCertifiedWeight(peer) => {
                write!(f, "missing or zero certified weight for cohort peer {peer:?}")
            }
            Self::UnsupportedCertifiedDifferentialPrivacy => write!(
                f,
                "certified Differential Privacy enforcement is unsupported in experimental accounting mode"
            ),
        }
    }
}

impl std::error::Error for RoundConfigError {}

impl RoundCoordinatorConfig {
    /// Validate configuration invariants (`ADR-0023` §1.1–1.3).
    pub fn validate(&self) -> Result<(), RoundConfigError> {
        if self.authorized_cohort.is_empty() {
            return Err(RoundConfigError::EmptyCohort);
        }
        if self.min_quorum == 0 || self.min_quorum > self.authorized_cohort.len() {
            return Err(RoundConfigError::InvalidQuorum {
                min_quorum: self.min_quorum,
                cohort_len: self.authorized_cohort.len(),
            });
        }
        if let UpdateKindV1::ModelDeltaV1 {
            server_learning_rate,
        } = self.update_kind
        {
            if !server_learning_rate.is_finite() || server_learning_rate <= 0.0 {
                return Err(RoundConfigError::InvalidServerLearningRate(
                    server_learning_rate,
                ));
            }
        }
        if let WeightingPolicyV1::RegistryCertified {
            ref peer_weights,
            max_weight_share,
        } = self.weighting_policy
        {
            if !matches!(self.aggregation, RobustAggregation::FedAvg) {
                return Err(RoundConfigError::IncompatibleWeightingPolicy);
            }
            if !max_weight_share.is_finite() || max_weight_share <= 0.0 || max_weight_share > 1.0 {
                return Err(RoundConfigError::InvalidMaxWeightShare(max_weight_share));
            }
            for peer in &self.authorized_cohort {
                match peer_weights.get(peer) {
                    Some(&w) if w > 0 => {}
                    _ => return Err(RoundConfigError::MissingOrZeroCertifiedWeight(*peer)),
                }
            }
        }
        Ok(())
    }
}

/// Unverified round submission DTO kept only as a legacy compatibility boundary (`ST-01`).
///
/// [`SwarmRoundCoordinator`] rejects unauthenticated [`PeerRoundSubmissionV1`] inputs fail-closed;
/// callers must deliver signed [`BundledGossipDeliveryV2`] payloads through the 3-stage pipeline
/// (`prepare_verified_delivery` -> [`ExpectedRoundContextV1`] -> [`VerifiedRoundSubmissionV1`]).
#[deprecated(
    note = "unverified ingress DTO; use prepare_verified_delivery -> ExpectedRoundContextV1 -> VerifiedRoundSubmissionV1"
)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PeerRoundSubmissionV1 {
    /// Run identifier this update is bound to.
    pub run_id: RunId,
    /// Target training round identifier.
    pub round_id: u64,
    /// Expected parent checkpoint content digest for `round_id`.
    pub base_checkpoint_digest: CheckpointContentDigestV1,
    /// Authenticated sender `PeerId`.
    pub sender: PeerId,
    /// Monotonic replay sequence number.
    pub sequence: u64,
    /// Sender timestamp in Unix seconds (for `ReplayProtection`).
    pub timestamp_secs: u64,
    /// Update kind (`ModelDeltaV1` or `GradientV1`).
    pub update_kind: UpdateKindV1,
    /// Flat parameter delta or gradient coordinates.
    pub values: Vec<f32>,
    /// Advisory wire-reported example count (logged for telemetry only; never trusted for weights).
    pub reported_examples: Option<u64>,
}

#[allow(deprecated)]
impl PeerRoundSubmissionV1 {
    /// Compute the canonical SHA-256 digest of this submission's round-bound payload.
    pub fn canonical_payload_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(PEER_SUBMISSION_DIGEST_DOMAIN_V1);
        hasher.update(self.run_id.as_bytes());
        hasher.update(self.round_id.to_le_bytes());
        hasher.update(self.base_checkpoint_digest.as_bytes());
        hasher.update(self.sender.as_bytes());
        let kind_tag: u8 = match self.update_kind {
            UpdateKindV1::ModelDeltaV1 { .. } => 1,
            UpdateKindV1::GradientV1 => 2,
        };
        hasher.update([kind_tag]);
        hasher.update((self.values.len() as u64).to_le_bytes());
        for &v in &self.values {
            hasher.update(v.to_bits().to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..]);
        out
    }
}

/// Compression schema version `0`: raw contiguous little-endian `f32` coordinates (`num_elements * 4` bytes).
pub const COMPRESSION_SCHEMA_DENSE_F32_LE_V1: u16 = 0;
/// Compression schema version `1`: JSON-serialized [`CompressedGradient`] (`swarm-torch-core`).
pub const COMPRESSION_SCHEMA_COMPRESSED_GRADIENT_V1: u16 = 1;
/// Compression schema version `2`: JSON-serialized [`LoraAdapterPairV1`] (`swarm-torch-models`).
pub const COMPRESSION_SCHEMA_LORA_PAIR_V1: u16 = 2;

const ROUND_SUBMISSION_DIGEST_DOMAIN_V2: &[u8] = b"swarmtorch.round.submission.v2";
const CONTEXT_POLICY_DIGEST_DOMAIN_V1: &[u8] = b"swarmtorch.coordinator.policy.v1";

/// Encode a dense `f32` slice into canonical little-endian bytes (`compression_schema_version = 0`).
pub fn encode_dense_update_payload_v2(values: &[f32]) -> Result<Vec<u8>, RoundError> {
    let mut out = Vec::with_capacity(values.len().saturating_mul(4));
    for (idx, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(RoundError::NonFiniteInput {
                sender: PeerId::new([0u8; 32]),
                coord_index: idx,
            });
        }
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// Encode a [`CompressedGradient`] for [`InnerUpdateV2`] (`compression_schema_version = 1`).
pub fn encode_compressed_update_payload_v2(
    compressed: &CompressedGradient,
) -> Result<Vec<u8>, RoundError> {
    compressed
        .validate_shape_and_bounds(swarm_torch_core::compression::DEFAULT_MAX_COMPRESSION_ELEMENTS)
        .map_err(|e| {
            RoundError::InvalidWirePayload(format!("invalid compressed gradient: {e:?}"))
        })?;
    serde_json::to_vec(compressed).map_err(|e| {
        RoundError::InvalidWirePayload(format!("failed to serialize compressed gradient: {e}"))
    })
}

/// Encode a validated [`LoraAdapterPairV1`] for [`InnerUpdateV2`] (`compression_schema_version = 2`).
pub fn encode_lora_update_payload_v2(pair: &LoraAdapterPairV1) -> Result<Vec<u8>, RoundError> {
    let _ = LoraAdapterPairV1::try_new(
        pair.descriptor.clone(),
        pair.d_in,
        pair.d_out,
        pair.a_weights.clone(),
        pair.b_weights.clone(),
    )
    .map_err(|e| RoundError::InvalidWirePayload(format!("invalid LoRA adapter pair: {e:?}")))?;
    serde_json::to_vec(pair).map_err(|e| {
        RoundError::InvalidWirePayload(format!("failed to serialize LoRA adapter pair: {e}"))
    })
}

#[allow(clippy::too_many_arguments)]
fn canonical_dense_submission_digest_v2(
    swarm_instance_id: &[u8; 16],
    run_id: RunId,
    round_id: u64,
    base_checkpoint_digest: &CheckpointContentDigestV1,
    context_policy_digest: &[u8; 32],
    sender: PeerId,
    payload_kind: PayloadKindV1,
    values: &[f32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(ROUND_SUBMISSION_DIGEST_DOMAIN_V2);
    hasher.update(swarm_instance_id);
    hasher.update(run_id.as_bytes());
    hasher.update(round_id.to_le_bytes());
    hasher.update(base_checkpoint_digest.as_bytes());
    hasher.update(context_policy_digest);
    hasher.update(sender.as_bytes());
    hasher.update([payload_kind as u8]);
    hasher.update((values.len() as u64).to_le_bytes());
    for &v in values {
        hasher.update(v.to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

fn check_update_kind_compatible(
    expected: UpdateKindV1,
    kind: PayloadKindV1,
) -> Result<(), RoundError> {
    match (expected, kind) {
        (UpdateKindV1::ModelDeltaV1 { .. }, PayloadKindV1::ModelDeltaV1)
        | (UpdateKindV1::GradientV1, PayloadKindV1::GradientV1) => Ok(()),
        (UpdateKindV1::ModelDeltaV1 { .. }, PayloadKindV1::GradientV1) => {
            Err(RoundError::UpdateKindMismatch {
                expected,
                got: UpdateKindV1::GradientV1,
            })
        }
        (UpdateKindV1::GradientV1, PayloadKindV1::ModelDeltaV1) => {
            Err(RoundError::UpdateKindMismatch {
                expected,
                got: UpdateKindV1::ModelDeltaV1 {
                    server_learning_rate: 1.0,
                },
            })
        }
        (_, PayloadKindV1::Ef21InnovationV1) => Err(RoundError::InvalidWirePayload(
            "EF21 innovation payload is unsupported on dense/LoRA coordinator ingress".to_string(),
        )),
    }
}

#[derive(Debug, Clone)]
enum PreparedPayloadV2 {
    Dense {
        values: Vec<f32>,
        canonical_payload_digest: [u8; 32],
    },
    Lora {
        pair: LoraAdapterPairV1,
    },
}

/// Stage A output: cryptographically authenticated, payload-decoded submission holding
/// a read-only [`PreparedReplayAdmission`] proof (`ST-01`, `ST-03`).
///
/// Producing [`PreparedSubmissionV1`] performs zero mutation to coordinator or replay state.
#[derive(Debug)]
pub struct PreparedSubmissionV1 {
    sender: PeerId,
    swarm_instance_id: [u8; 16],
    run_id: RunId,
    round_id: u64,
    base_checkpoint_digest: CheckpointContentDigestV1,
    context_policy_digest: [u8; 32],
    payload_kind: PayloadKindV1,
    reported_examples: Option<u64>,
    payload: PreparedPayloadV2,
    replay_admission: PreparedReplayAdmission,
}

impl PreparedSubmissionV1 {
    /// Authenticated sender `PeerId` derived from Ed25519 public key.
    pub const fn sender(&self) -> PeerId {
        self.sender
    }

    /// Swarm deployment/cohort instance identifier carried inside the signed inner update.
    pub const fn swarm_instance_id(&self) -> [u8; 16] {
        self.swarm_instance_id
    }

    /// Run identifier bound inside both the wrapper and inner update.
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Round identifier bound inside both the wrapper and inner update.
    pub const fn round_id(&self) -> u64 {
        self.round_id
    }

    /// Parent checkpoint content digest bound inside the signed inner update.
    pub const fn base_checkpoint_digest(&self) -> CheckpointContentDigestV1 {
        self.base_checkpoint_digest
    }

    /// Context/aggregation policy digest bound inside the signed inner update.
    pub const fn context_policy_digest(&self) -> [u8; 32] {
        self.context_policy_digest
    }

    /// Payload kind discriminator bound inside the signed inner update.
    pub const fn payload_kind(&self) -> PayloadKindV1 {
        self.payload_kind
    }

    /// Advisory reported local example count (telemetry only).
    pub const fn reported_examples(&self) -> Option<u64> {
        self.reported_examples
    }

    /// Authenticated delivery sequence number.
    pub fn sequence(&self) -> u64 {
        self.replay_admission.sequence()
    }

    /// Authenticated delivery timestamp in Unix seconds.
    pub fn timestamp_secs(&self) -> u32 {
        self.replay_admission.timestamp()
    }

    /// Receiver-bound v2 session identifier validated during Stage A.
    pub fn session_id(&self) -> Option<[u8; 32]> {
        self.replay_admission.session_id()
    }
}

/// Immutable coordinator context snapshot used in Stage B to validate [`PreparedSubmissionV1`]
/// with zero mutation to coordinator or replay state (`ST-01`).
#[derive(Debug, Clone)]
pub struct ExpectedRoundContextV1 {
    swarm_instance_id: [u8; 16],
    coordinator_peer_id: PeerId,
    run_id: RunId,
    active_round_id: u64,
    max_staleness: u64,
    accepted_parent_digest: CheckpointContentDigestV1,
    historical_digests_snapshot: BTreeMap<u64, CheckpointContentDigestV1>,
    historical_anchor_rounds: BTreeSet<u64>,
    expected_update_kind: UpdateKindV1,
    expected_param_count: usize,
    cohort_allowlist_snapshot: BTreeSet<PeerId>,
    equivocated_snapshot: BTreeSet<PeerId>,
    context_policy_digest: [u8; 32],
    fedasync_policy: Option<swarm_torch_core::aggregation::FedAsyncStalenessDescriptorV1>,
    coordinator_generation: u64,
}

impl ExpectedRoundContextV1 {
    /// Expected swarm instance identifier.
    pub const fn swarm_instance_id(&self) -> [u8; 16] {
        self.swarm_instance_id
    }

    /// Coordinator local recipient `PeerId`.
    pub const fn coordinator_peer_id(&self) -> PeerId {
        self.coordinator_peer_id
    }

    /// Bound training/coordination run identifier.
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Active target round identifier when this snapshot was taken.
    pub const fn active_round_id(&self) -> u64 {
        self.active_round_id
    }

    /// Active checkpoint digest expected for `active_round_id`.
    pub const fn accepted_parent_digest(&self) -> CheckpointContentDigestV1 {
        self.accepted_parent_digest
    }

    /// Context/aggregation policy digest expected on signed v2 updates.
    pub const fn context_policy_digest(&self) -> [u8; 32] {
        self.context_policy_digest
    }

    /// Monotonic coordinator generation when this context was exported.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    fn validate_common_context(&self, prepared: &PreparedSubmissionV1) -> Result<(), RoundError> {
        if prepared.swarm_instance_id != self.swarm_instance_id {
            return Err(RoundError::SwarmInstanceMismatch {
                expected: self.swarm_instance_id,
                got: prepared.swarm_instance_id,
            });
        }
        if prepared.run_id != self.run_id {
            return Err(RoundError::RunIdMismatch {
                expected: self.run_id,
                got: prepared.run_id,
            });
        }
        if prepared.context_policy_digest != self.context_policy_digest {
            return Err(RoundError::ContextPolicyDigestMismatch {
                expected: self.context_policy_digest,
                got: prepared.context_policy_digest,
            });
        }
        if !self.cohort_allowlist_snapshot.contains(&prepared.sender) {
            return Err(RoundError::UnauthorizedPeer(prepared.sender));
        }
        if !prepared.replay_admission.is_v2() || prepared.replay_admission.session_id().is_none() {
            return Err(RoundError::ReplayRejected(ReplayError::RecoveryRequired {
                peer: prepared.sender,
                current_epoch: 0,
                requested_epoch: 1,
            }));
        }
        Ok(())
    }

    /// Stage B (synchronous/bounded-staleness round): validate `prepared` against this immutable
    /// round context with zero state mutation (`ST-01`).
    pub fn validate_sync_submission(
        &self,
        prepared: PreparedSubmissionV1,
    ) -> Result<ValidatedRoundSubmissionV1, RoundError> {
        self.validate_common_context(&prepared)?;

        if prepared.round_id > self.active_round_id {
            return Err(RoundError::FutureRoundRejected {
                submission_round: prepared.round_id,
                current_round: self.active_round_id,
            });
        }
        let staleness = self.active_round_id - prepared.round_id;
        if staleness > self.max_staleness {
            return Err(RoundError::StaleRoundRejected {
                submission_round: prepared.round_id,
                current_round: self.active_round_id,
                max_staleness: self.max_staleness,
            });
        }

        let expected_base = if staleness == 0 {
            self.accepted_parent_digest
        } else {
            let prev_round = prepared.round_id.saturating_sub(1);
            self.historical_digests_snapshot
                .get(&prev_round)
                .copied()
                .ok_or(RoundError::StaleAnchorEvicted {
                    base_round: prev_round,
                    current_round: self.active_round_id,
                })?
        };
        if prepared.base_checkpoint_digest != expected_base {
            return Err(RoundError::BaseCheckpointMismatch {
                expected: expected_base,
                got: prepared.base_checkpoint_digest,
            });
        }

        check_update_kind_compatible(self.expected_update_kind, prepared.payload_kind)?;

        if self.equivocated_snapshot.contains(&prepared.sender) {
            return Err(RoundError::EquivocationRejected {
                sender: prepared.sender,
                round_id: self.active_round_id,
            });
        }

        let (values, canonical_payload_digest) = match prepared.payload {
            PreparedPayloadV2::Dense {
                values,
                canonical_payload_digest,
            } => (values, canonical_payload_digest),
            PreparedPayloadV2::Lora { .. } => {
                return Err(RoundError::InvalidWirePayload(
                    "LoRA adapter payload must use validate_lora_submission".to_string(),
                ));
            }
        };
        if values.len() != self.expected_param_count {
            return Err(RoundError::DimensionMismatch {
                expected: self.expected_param_count,
                got: values.len(),
            });
        }

        Ok(ValidatedRoundSubmissionV1 {
            coordinator_generation: self.coordinator_generation,
            target_round_id: self.active_round_id,
            sender: prepared.sender,
            canonical_payload_digest,
            values,
            reported_examples: prepared.reported_examples,
            replay_admission: prepared.replay_admission,
        })
    }

    /// Stage B (`FedAsync`): validate a bounded-staleness asynchronous update against retained
    /// historical anchors with zero state mutation (`ST-01`).
    pub fn validate_fedasync_submission(
        &self,
        prepared: PreparedSubmissionV1,
    ) -> Result<ValidatedFedAsyncSubmissionV1, RoundError> {
        let policy = self
            .fedasync_policy
            .ok_or(RoundError::Aggregation(CoreError::InvalidThreshold))?;
        self.validate_common_context(&prepared)?;

        if !matches!(self.expected_update_kind, UpdateKindV1::ModelDeltaV1 { .. })
            || prepared.payload_kind != PayloadKindV1::ModelDeltaV1
        {
            check_update_kind_compatible(self.expected_update_kind, prepared.payload_kind)?;
            return Err(RoundError::UpdateKindMismatch {
                expected: self.expected_update_kind,
                got: UpdateKindV1::GradientV1,
            });
        }

        let latest_committed = self.active_round_id.saturating_sub(1);
        if prepared.round_id > latest_committed {
            return Err(RoundError::FutureRoundRejected {
                submission_round: prepared.round_id,
                current_round: latest_committed,
            });
        }
        let staleness = latest_committed - prepared.round_id;
        let max_stale = policy.max_staleness_rounds.min(self.max_staleness);
        if staleness > max_stale {
            return Err(RoundError::StaleRoundRejected {
                submission_round: prepared.round_id,
                current_round: latest_committed,
                max_staleness: max_stale,
            });
        }

        let expected_base = self
            .historical_digests_snapshot
            .get(&prepared.round_id)
            .copied()
            .filter(|_| self.historical_anchor_rounds.contains(&prepared.round_id))
            .ok_or(RoundError::StaleAnchorEvicted {
                base_round: prepared.round_id,
                current_round: latest_committed,
            })?;
        if prepared.base_checkpoint_digest != expected_base {
            return Err(RoundError::BaseCheckpointMismatch {
                expected: expected_base,
                got: prepared.base_checkpoint_digest,
            });
        }

        let values = match prepared.payload {
            PreparedPayloadV2::Dense { values, .. } => values,
            PreparedPayloadV2::Lora { .. } => {
                return Err(RoundError::InvalidWirePayload(
                    "LoRA adapter payload is unsupported in dense FedAsync step".to_string(),
                ));
            }
        };
        if values.len() != self.expected_param_count {
            return Err(RoundError::DimensionMismatch {
                expected: self.expected_param_count,
                got: values.len(),
            });
        }

        Ok(ValidatedFedAsyncSubmissionV1 {
            coordinator_generation: self.coordinator_generation,
            base_round_id: prepared.round_id,
            sender: prepared.sender,
            values,
            replay_admission: prepared.replay_admission,
        })
    }

    /// Stage B (LoRA/FLoRA cohort): validate a prepared LoRA adapter pair against the active round
    /// context with zero state mutation (`ST-01`).
    pub fn validate_lora_submission(
        &self,
        prepared: PreparedSubmissionV1,
    ) -> Result<ValidatedLoraSubmissionV1, RoundError> {
        self.validate_common_context(&prepared)?;

        if prepared.round_id > self.active_round_id {
            return Err(RoundError::FutureRoundRejected {
                submission_round: prepared.round_id,
                current_round: self.active_round_id,
            });
        }
        if prepared.round_id < self.active_round_id {
            return Err(RoundError::StaleRoundRejected {
                submission_round: prepared.round_id,
                current_round: self.active_round_id,
                max_staleness: 0,
            });
        }
        if prepared.base_checkpoint_digest != self.accepted_parent_digest {
            return Err(RoundError::BaseCheckpointMismatch {
                expected: self.accepted_parent_digest,
                got: prepared.base_checkpoint_digest,
            });
        }
        if prepared.payload_kind != PayloadKindV1::ModelDeltaV1 {
            return Err(RoundError::InvalidWirePayload(
                "LoRA adapter submissions require PayloadKindV1::ModelDeltaV1".to_string(),
            ));
        }

        let pair = match prepared.payload {
            PreparedPayloadV2::Lora { pair, .. } => pair,
            PreparedPayloadV2::Dense { .. } => {
                return Err(RoundError::InvalidWirePayload(
                    "dense vector payload cannot enter validate_lora_submission".to_string(),
                ));
            }
        };

        Ok(ValidatedLoraSubmissionV1 {
            coordinator_generation: self.coordinator_generation,
            round_id: prepared.round_id,
            sender: prepared.sender,
            pair,
            replay_admission: prepared.replay_admission,
        })
    }
}

/// Stage B output for synchronous round admission (private fields, read-only before Stage C).
#[derive(Debug)]
pub struct ValidatedRoundSubmissionV1 {
    coordinator_generation: u64,
    target_round_id: u64,
    sender: PeerId,
    canonical_payload_digest: [u8; 32],
    values: Vec<f32>,
    reported_examples: Option<u64>,
    replay_admission: PreparedReplayAdmission,
}

impl ValidatedRoundSubmissionV1 {
    /// Coordinator generation stamped during Stage B validation.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    /// Verified originator `PeerId`.
    pub const fn sender(&self) -> PeerId {
        self.sender
    }

    /// Target coordinator round identifier.
    pub const fn target_round_id(&self) -> u64 {
        self.target_round_id
    }
}

/// Stage B output for `FedAsync` admission (private fields, read-only before Stage C).
#[derive(Debug)]
pub struct ValidatedFedAsyncSubmissionV1 {
    coordinator_generation: u64,
    base_round_id: u64,
    sender: PeerId,
    values: Vec<f32>,
    replay_admission: PreparedReplayAdmission,
}

impl ValidatedFedAsyncSubmissionV1 {
    /// Coordinator generation stamped during Stage B validation.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    /// Verified originator `PeerId`.
    pub const fn sender(&self) -> PeerId {
        self.sender
    }

    /// Historical anchor base round identifier.
    pub const fn base_round_id(&self) -> u64 {
        self.base_round_id
    }
}

/// Stage B output for LoRA cohort admission (private fields, read-only before Stage C).
#[derive(Debug)]
pub struct ValidatedLoraSubmissionV1 {
    coordinator_generation: u64,
    round_id: u64,
    sender: PeerId,
    pair: LoraAdapterPairV1,
    replay_admission: PreparedReplayAdmission,
}

impl ValidatedLoraSubmissionV1 {
    /// Coordinator generation stamped during Stage B validation.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    /// Verified originator `PeerId`.
    pub const fn sender(&self) -> PeerId {
        self.sender
    }

    /// Target round identifier.
    pub const fn round_id(&self) -> u64 {
        self.round_id
    }
}

/// Stage C sealed token proving replay sequence consumption and round context verification (`ST-01`).
#[derive(Debug)]
pub struct VerifiedRoundSubmissionV1 {
    coordinator_generation: u64,
    round_id: u64,
    sender: PeerId,
    sequence: u64,
    canonical_payload_digest: [u8; 32],
    values: Vec<f32>,
    reported_examples: Option<u64>,
}

impl VerifiedRoundSubmissionV1 {
    /// Coordinator generation stamped when this token was committed.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    /// Verified originator `PeerId`.
    pub const fn sender(&self) -> PeerId {
        self.sender
    }

    /// Verified target round identifier.
    pub const fn round_id(&self) -> u64 {
        self.round_id
    }

    /// Consumed replay sequence number.
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }
}

/// Stage C sealed token for a verified `FedAsync` update (`ST-01`).
#[derive(Debug)]
pub struct VerifiedFedAsyncSubmissionV1 {
    coordinator_generation: u64,
    base_round_id: u64,
    sender: PeerId,
    values: Vec<f32>,
}

impl VerifiedFedAsyncSubmissionV1 {
    /// Coordinator generation stamped when this token was committed.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    /// Verified originator `PeerId`.
    pub const fn sender(&self) -> PeerId {
        self.sender
    }

    /// Historical anchor base round identifier.
    pub const fn base_round_id(&self) -> u64 {
        self.base_round_id
    }
}

/// Stage C sealed token for a verified LoRA cohort contribution (`ST-01`).
#[derive(Debug)]
pub struct VerifiedLoraSubmissionV1 {
    coordinator_generation: u64,
    round_id: u64,
    sender: PeerId,
    pair: LoraAdapterPairV1,
}

impl VerifiedLoraSubmissionV1 {
    /// Coordinator generation stamped when this token was committed.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    /// Verified originator `PeerId`.
    pub const fn sender(&self) -> PeerId {
        self.sender
    }

    /// Target round identifier.
    pub const fn round_id(&self) -> u64 {
        self.round_id
    }
}

/// Outcome of submitting an update to [`SwarmRoundCoordinator`] (`ADR-0023` §1.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmissionDecision {
    /// First valid submission from `sender` for the round; buffered for aggregation.
    Admitted {
        admitted_count: usize,
        min_quorum: usize,
    },
    /// Byte-identical re-transmission from `sender` for the same round; acknowledged idempotently
    /// without double-counting the peer's vote.
    DuplicateIdempotentAck,
}

/// Errors returned when a peer submission or round finalization is rejected (`ADR-0023` §1).
#[derive(Debug)]
pub enum RoundError {
    Config(RoundConfigError),
    RunIdMismatch {
        expected: RunId,
        got: RunId,
    },
    FutureRoundRejected {
        submission_round: u64,
        current_round: u64,
    },
    StaleRoundRejected {
        submission_round: u64,
        current_round: u64,
        max_staleness: u64,
    },
    BaseCheckpointMismatch {
        expected: CheckpointContentDigestV1,
        got: CheckpointContentDigestV1,
    },
    UpdateKindMismatch {
        expected: UpdateKindV1,
        got: UpdateKindV1,
    },
    UnauthorizedPeer(PeerId),
    EquivocationRejected {
        sender: PeerId,
        round_id: u64,
    },
    ReplayRejected(ReplayError),
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    NonFiniteInput {
        sender: PeerId,
        coord_index: usize,
    },
    InsufficientQuorum {
        admitted: usize,
        min_quorum: usize,
    },
    Aggregation(CoreError),
    OptimizerStep(String),
    FitnessEvaluation(String),
    PrivacyBudget(swarm_torch_core::crypto::CryptoError),
    StaleAnchorEvicted {
        base_round: u64,
        current_round: u64,
    },
    /// Unverified or deprecated ingress path rejected fail-closed (`ST-01`).
    UnverifiedSubmissionRejected,
    /// Wire decoding or Ed25519 signature verification failed (`ST-01`, `ST-03`).
    AuthenticationFailed(GossipVerifyError),
    /// Wire payload `swarm_instance_id` did not match coordinator instance (`ST-01`).
    SwarmInstanceMismatch {
        expected: [u8; 16],
        got: [u8; 16],
    },
    /// Wire payload `context_policy_digest` did not match active coordinator policy (`ST-01`).
    ContextPolicyDigestMismatch {
        expected: [u8; 32],
        got: [u8; 32],
    },
    /// Sealed submission token belongs to a stale or mismatched coordinator generation (`ST-01`).
    StaleCoordinatorGeneration {
        expected: u64,
        got: u64,
    },
    /// Wire payload compression/schema/LoRA decoding failed (`ST-01`).
    InvalidWirePayload(String),
    /// Round identifier reached `u64::MAX` and cannot advance safely (`ST-04`).
    RoundIdOverflow {
        round_id: u64,
    },
    /// Coordinator generation counter reached `u64::MAX` (`ST-04`).
    CoordinatorGenerationOverflow {
        generation: u64,
    },
    /// Required resume state dependency (optimizer state, CClip center, anchors, or manifest)
    /// is missing, mismatched, or incompatible (`ST-04`).
    ResumeStateMismatch(String),
    /// Optimizer state snapshot serialization or restoration failed (`ST-04`).
    OptimizerStateError(String),
    Io(io::Error),
}

impl core::fmt::Display for RoundError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Config(e) => write!(f, "invalid round coordinator config: {e}"),
            Self::RunIdMismatch { expected, got } => {
                write!(f, "run_id mismatch: expected {expected}, got {got}")
            }
            Self::FutureRoundRejected {
                submission_round,
                current_round,
            } => write!(
                f,
                "future round submission rejected: submission round {submission_round} > current round {current_round}"
            ),
            Self::StaleRoundRejected {
                submission_round,
                current_round,
                max_staleness,
            } => write!(
                f,
                "stale round submission rejected: submission round {submission_round}, current round {current_round}, max_staleness {max_staleness}"
            ),
            Self::BaseCheckpointMismatch { expected, got } => write!(
                f,
                "base_checkpoint_digest mismatch: expected {expected}, got {got}"
            ),
            Self::UpdateKindMismatch { expected, got } => {
                write!(f, "update_kind mismatch: expected {expected:?}, got {got:?}")
            }
            Self::UnauthorizedPeer(peer) => write!(f, "unauthorized cohort peer: {peer:?}"),
            Self::EquivocationRejected { sender, round_id } => write!(
                f,
                "equivocation detected from peer {sender:?} in round {round_id}; peer vote revoked"
            ),
            Self::ReplayRejected(e) => write!(f, "replay protection rejected submission: {e}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "parameter dimension mismatch: expected {expected}, got {got}")
            }
            Self::NonFiniteInput {
                sender,
                coord_index,
            } => write!(
                f,
                "non-finite value at coordinate {coord_index} from peer {sender:?}"
            ),
            Self::InsufficientQuorum {
                admitted,
                min_quorum,
            } => write!(
                f,
                "insufficient quorum to finalize round: admitted {admitted} < min_quorum {min_quorum}"
            ),
            Self::Aggregation(e) => write!(f, "aggregation failed: {e}"),
            Self::OptimizerStep(msg) => write!(f, "server optimizer step failed: {msg}"),
            Self::FitnessEvaluation(msg) => write!(f, "fitness evaluation failed: {msg}"),
            Self::PrivacyBudget(e) => write!(f, "differential privacy budget error: {e}"),
            Self::StaleAnchorEvicted {
                base_round,
                current_round,
            } => write!(
                f,
                "stale base round {base_round} evicted from historical anchor window at round {current_round}"
            ),
            Self::UnverifiedSubmissionRejected => write!(
                f,
                "unverified submission rejected: use authenticated BundledGossipDeliveryV2 3-stage pipeline"
            ),
            Self::AuthenticationFailed(e) => write!(
                f,
                "gossip authentication or wire envelope verification failed: {e:?}"
            ),
            Self::SwarmInstanceMismatch { expected, got } => write!(
                f,
                "swarm_instance_id mismatch: expected {expected:02x?}, got {got:02x?}"
            ),
            Self::ContextPolicyDigestMismatch { expected, got } => write!(
                f,
                "context_policy_digest mismatch: expected {expected:02x?}, got {got:02x?}"
            ),
            Self::StaleCoordinatorGeneration { expected, got } => write!(
                f,
                "stale coordinator_generation on verified submission: expected {expected}, got {got}"
            ),
            Self::InvalidWirePayload(msg) => write!(f, "invalid wire payload: {msg}"),
            Self::RoundIdOverflow { round_id } => {
                write!(f, "round_id overflow advancing past {round_id}")
            }
            Self::CoordinatorGenerationOverflow { generation } => {
                write!(f, "coordinator_generation overflow advancing past {generation}")
            }
            Self::ResumeStateMismatch(msg) => write!(f, "resume state validation failed: {msg}"),
            Self::OptimizerStateError(msg) => write!(f, "optimizer state error: {msg}"),
            Self::Io(e) => write!(f, "artifact/CAS I/O error: {e}"),
        }
    }
}

impl std::error::Error for RoundError {}

impl From<GossipVerifyError> for RoundError {
    fn from(e: GossipVerifyError) -> Self {
        match e {
            GossipVerifyError::Replay(replay_err) => Self::ReplayRejected(replay_err),
            other => Self::AuthenticationFailed(other),
        }
    }
}

impl From<io::Error> for RoundError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<RoundConfigError> for RoundError {
    fn from(e: RoundConfigError) -> Self {
        Self::Config(e)
    }
}

impl From<CoreError> for RoundError {
    fn from(e: CoreError) -> Self {
        Self::Aggregation(e)
    }
}

#[derive(Debug, Clone)]
struct AdmittedPeerEntry {
    payload_digest: [u8; 32],
    update: GradientUpdate,
    reported_examples: Option<u64>,
}

/// Summary of a durably committed training round (`ADR-0023` §1, `W10-01`).
#[derive(Debug, Clone)]
pub struct RoundCommitSummaryV1 {
    pub round_id: u64,
    pub admitted_peers: Vec<PeerId>,
    pub equivocated_peers: Vec<PeerId>,
    pub checkpoint_asset_key: String,
    pub recipe_key: CheckpointRecipeKeyV1,
    pub content_digest: CheckpointContentDigestV1,
    pub blob_uri: String,
    pub fitness_score: Option<f32>,
    /// Immutable snapshot of each admitted peer's raw pre-transform update vector.
    pub frozen_raw_inputs: BTreeMap<PeerId, Vec<f32>>,
    /// Warnings emitted when a peer's advisory `reported_examples` diverged from `RegistryCertified` weight.
    pub weight_mismatch_warnings: Vec<String>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum CommitFailPoint {
    AfterCasBeforeManifest,
    AfterManifestBeforeCommittedHead,
    AfterCommittedHeadBeforeMemory,
}

/// Unified round coordinator state machine (`ADR-0023` §1, `W10-01`).
pub struct SwarmRoundCoordinator<M: SwarmModel> {
    run_id: RunId,
    swarm_instance_id: [u8; 16],
    coordinator_peer_id: PeerId,
    coordinator_generation: u64,
    config: RoundCoordinatorConfig,
    current_round: u64,
    model: M,
    parent_digest: CheckpointContentDigestV1,
    parent_recipe_key: CheckpointRecipeKeyV1,
    parent_asset_key: String,
    historical_digests: BTreeMap<u64, CheckpointContentDigestV1>,
    replay: ReplayProtection,
    admitted: BTreeMap<PeerId, AdmittedPeerEntry>,
    equivocated: BTreeSet<PeerId>,
    /// Persistent previous-round aggregate center $v^{(r-1)}$ for [`CenteredClipping`] (`ADR-0019` §4).
    cclip_center: Option<Vec<f32>>,
    /// Optional `$s$-Bucketing` pre-aggregation configuration (`ADR-0019` §4).
    bucketing: Option<BucketingConfig>,
    /// Execution report from the most recent `$s$-Bucketing` step.
    last_bucketing_report: Option<BucketExecutionReport>,
    /// Optional stateful Rényi Differential Privacy accountant (`ADR-0023` §4.2, `W12-03`).
    dp_accountant: Option<swarm_torch_core::dp::RdpAccountantV1>,
    /// Optional bounded-staleness asynchronous policy (`ADR-0023` §4.1, `W12-03`).
    fedasync_policy: Option<swarm_torch_core::aggregation::FedAsyncStalenessDescriptorV1>,
    /// Optional hinge threshold (in rounds) before staleness decay begins.
    fedasync_hinge_threshold: u64,
    /// Optional pre-drift $\ell_2$ clipping threshold $\tau$ for `FedAsync` deltas.
    fedasync_clip_norm: Option<f32>,
    /// Bounded historical parameter anchor window (`round_id -> w_round`) for `FedAsync` drift compensation.
    historical_anchors: BTreeMap<u64, Vec<f32>>,
    /// Optional Ed25519 keypair used to sign every round's Step 3 manifest before committing `committed_head.json`.
    manifest_signer: Option<KeyPair>,
    /// Deterministic test fault-injection seam across the 5-step transactional commit boundary (`ST-04`).
    #[cfg(test)]
    fail_point: Option<CommitFailPoint>,
}

struct StagedRoundCommit<M> {
    committed_round: u64,
    next_round: u64,
    next_generation: u64,
    candidate_model: M,
    optimizer_snapshot: Option<OptimizerStateSnapshotV1>,
    next_cclip_center: Option<Vec<f32>>,
    next_historical_anchors: BTreeMap<u64, Vec<f32>>,
    next_historical_digests_without_new: BTreeMap<u64, CheckpointContentDigestV1>,
    next_bucketing_report: Option<BucketExecutionReport>,
    checkpoint_asset_key: String,
    node_key: String,
    op_type: String,
    output_rows: Option<u64>,
    admitted_peers: Vec<PeerId>,
    equivocated_peers: Vec<PeerId>,
    frozen_raw_inputs: BTreeMap<PeerId, Vec<f32>>,
    weight_mismatch_warnings: Vec<String>,
    fitness_score: Option<f32>,
    staged_transform_audits: Vec<TransformAuditV0>,
    dp_subsampling_q_override: Option<f64>,
    dp_explicit_tau: Option<f32>,
    emit_standard_telemetry: bool,
    clear_round_admitted: bool,
    ts_unix_nanos: u64,
    telemetry_admitted_count: usize,
    telemetry_equivocated_count: usize,
}

fn hex_lower_32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        use core::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn hash_round_coordinator_config(hasher: &mut Sha256, config: &RoundCoordinatorConfig) {
    match config.update_kind {
        UpdateKindV1::ModelDeltaV1 {
            server_learning_rate,
        } => {
            hasher.update([1u8]);
            hasher.update(server_learning_rate.to_bits().to_le_bytes());
        }
        UpdateKindV1::GradientV1 => hasher.update([2u8]),
    }

    match &config.aggregation {
        RobustAggregation::FedAvg => hasher.update([1u8]),
        RobustAggregation::Median => hasher.update([2u8]),
        RobustAggregation::TrimmedMean { trim_ratio } => {
            hasher.update([3u8]);
            hasher.update(trim_ratio.to_bits().to_le_bytes());
        }
        #[cfg(feature = "robust-aggregation")]
        RobustAggregation::Krum { num_byzantine } => {
            hasher.update([4u8]);
            hasher.update((*num_byzantine as u64).to_le_bytes());
        }
        #[cfg(feature = "robust-aggregation")]
        RobustAggregation::MultiKrum {
            num_byzantine,
            num_selected,
        } => {
            hasher.update([5u8]);
            hasher.update((*num_byzantine as u64).to_le_bytes());
            hasher.update((*num_selected as u64).to_le_bytes());
        }
        #[cfg(feature = "robust-aggregation")]
        RobustAggregation::Bulyan {
            max_byzantine,
            profile,
        } => {
            hasher.update([6u8]);
            hasher.update((*max_byzantine as u64).to_le_bytes());
            let p_tag: u8 = match profile {
                BulyanProfile::IterativeKrumV1 => 1,
            };
            hasher.update([p_tag]);
        }
        RobustAggregation::CenteredClipping {
            tau,
            max_iterations,
        } => {
            hasher.update([7u8]);
            hasher.update(tau.to_bits().to_le_bytes());
            hasher.update((*max_iterations as u64).to_le_bytes());
        }
    }

    match &config.weighting_policy {
        WeightingPolicyV1::EqualPeerVote => hasher.update([1u8]),
        WeightingPolicyV1::RegistryCertified {
            peer_weights,
            max_weight_share,
        } => {
            hasher.update([2u8]);
            hasher.update(max_weight_share.to_bits().to_le_bytes());
            hasher.update((peer_weights.len() as u64).to_le_bytes());
            for (peer, &weight) in peer_weights {
                hasher.update(peer.as_bytes());
                hasher.update(weight.to_le_bytes());
            }
        }
    }

    hasher.update((config.authorized_cohort.len() as u64).to_le_bytes());
    for peer in &config.authorized_cohort {
        hasher.update(peer.as_bytes());
    }
    hasher.update((config.min_quorum as u64).to_le_bytes());
    hasher.update(config.max_staleness.to_le_bytes());
    hasher.update(config.max_clock_skew_secs.to_le_bytes());
}

fn compute_base_config_policy_digest_hex(
    config: &RoundCoordinatorConfig,
    param_count: usize,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"swarmtorch.coordinator.base_config.v1");
    hasher.update((param_count as u64).to_le_bytes());
    hash_round_coordinator_config(&mut hasher, config);
    let digest: [u8; 32] = hasher.finalize().into();
    hex_lower_32(&digest)
}

fn default_swarm_instance_id(run_id: RunId) -> [u8; 16] {
    let digest = sha256_slices(&[b"swarmtorch.coordinator.instance.v1", run_id.as_bytes()]);
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

fn default_coordinator_peer_id(run_id: RunId) -> PeerId {
    PeerId::new(sha256_slices(&[
        b"swarmtorch.coordinator.peer.v1",
        run_id.as_bytes(),
    ]))
}

impl<M: SwarmModel + Clone> SwarmRoundCoordinator<M> {
    fn build_replay_protection(
        config: &RoundCoordinatorConfig,
        floor_secs: u32,
    ) -> ReplayProtection {
        let allowlist = CohortAllowlist::from_peers(config.authorized_cohort.iter().copied());
        let replay = ReplayProtection::try_with_config(1024, config.max_clock_skew_secs)
            .expect("non-zero capacity")
            .with_cohort_allowlist(allowlist);
        if floor_secs > 0 {
            replay.with_timestamp_floor(floor_secs)
        } else {
            replay
        }
    }

    fn compute_next_historical_anchors(
        &self,
        committed_round: u64,
        committed_weights: Vec<f32>,
    ) -> BTreeMap<u64, Vec<f32>> {
        let mut next = self.historical_anchors.clone();
        next.insert(committed_round, committed_weights);
        let max_keep = self.config.max_staleness.max(
            self.fedasync_policy
                .map(|p| p.max_staleness_rounds)
                .unwrap_or(0),
        );
        let cutoff = committed_round.saturating_sub(max_keep);
        next.retain(|&r, _| r >= cutoff);
        next
    }

    /// Initialize a new [`SwarmRoundCoordinator`] at round 1, persisting the genesis (round 0)
    /// model checkpoint, anchors, `CoordinatorResumeStateV2` CAS spine, finalized manifest, and
    /// authoritative `checkpoints/committed_head.json` (`ADR-0023` §1, §2.2, `ST-04`).
    pub fn initialize(
        run_id: RunId,
        config: RoundCoordinatorConfig,
        initial_model: M,
        session: &mut DataOpsSession,
        ts_unix_nanos: u64,
    ) -> Result<Self, RoundError> {
        config.validate()?;
        if initial_model.num_parameters() == 0 {
            return Err(RoundError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let (genesis_digest, genesis_uri, genesis_bytes) =
            put_checkpoint_model(&run_dir, &initial_model)?;
        let genesis_recipe_key = CheckpointRecipeKeyV1::genesis(run_id);
        let genesis_asset_key = "checkpoint://round/0".to_string();

        let ingest_node = NodeV1 {
            node_key: "train/round_0_genesis".to_string(),
            node_id: Some(node_id_from_key("train/round_0_genesis")),
            op_kind: OpKind::Train,
            op_type: "checkpoint_genesis".to_string(),
            inputs: vec![],
            outputs: vec![AssetRefV1 {
                asset_key: genesis_asset_key.clone(),
                fingerprint: Some(genesis_digest.to_hex()),
            }],
            params: CanonParams::new(),
            code_ref: Some(format!("swarm-torch@{}", env!("CARGO_PKG_VERSION"))),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
            execution_hint: None,
            cache_policy: None,
            materialization_policy: None,
            resources: None,
            op_hash: None,
        };

        session.register_source(
            &genesis_asset_key,
            TrustClass::Trusted,
            SourceDescriptorV0 {
                uri: genesis_uri.clone(),
                content_type: "application/x-swarmtorch-model-v1".to_string(),
                auth_mode: AuthModeMarker::None,
                etag_or_version: Some(genesis_digest.to_hex()),
            },
            None,
            &ingest_node,
        )?;

        let mut historical_digests = BTreeMap::new();
        historical_digests.insert(0, genesis_digest);
        let mut historical_anchors = BTreeMap::new();
        historical_anchors.insert(0, initial_model.parameters().to_vec());
        let (anchors_digest, anchors_uri, anchors_bytes) =
            put_historical_anchors_v1(&run_dir, &historical_anchors)?;

        let swarm_instance_id = default_swarm_instance_id(run_id);
        let coordinator_peer_id = default_coordinator_peer_id(run_id);
        let config_policy_digest_hex =
            compute_base_config_policy_digest_hex(&config, initial_model.num_parameters());

        let resume_v2 = CoordinatorResumeStateV2 {
            schema_version: COORDINATOR_RESUME_STATE_SCHEMA_V2,
            run_id,
            swarm_instance_id,
            coordinator_peer_id,
            generation: 0,
            round_id: 0,
            next_round_id: 1,
            checkpoint_asset_key: genesis_asset_key.clone(),
            recipe_key: genesis_recipe_key,
            model_digest: genesis_digest,
            model_blob_uri: genesis_uri.clone(),
            model_blob_bytes: genesis_bytes,
            parameter_count: initial_model.num_parameters(),
            optimizer_state_digest: None,
            optimizer_state_cas_uri: None,
            optimizer_state_bytes: None,
            cclip_center_digest: None,
            cclip_center_cas_uri: None,
            cclip_center_bytes: None,
            historical_anchors_digest: anchors_digest,
            historical_anchors_cas_uri: anchors_uri,
            historical_anchors_bytes: anchors_bytes,
            historical_digests: historical_digests.clone(),
            replay_timestamp_floor_secs: 0,
            config_policy_digest_hex,
            fedasync_policy: None,
            fedasync_hinge_threshold: 0,
            fedasync_clip_norm: None,
            bucketing: None,
            committed_unix_nanos: ts_unix_nanos,
        };
        let (_resume_v2_digest, resume_v2_uri, _resume_v2_bytes) =
            put_coordinator_resume_state_v2(&run_dir, &resume_v2)?;

        let (_manifest, manifest_sha256) = session.sink().finalize_checkpoint_manifest(0, None)?;
        let head = CommittedHeadV1 {
            schema_version: COMMITTED_HEAD_SCHEMA_V1,
            run_id,
            generation: 0,
            round_id: 0,
            manifest_sha256,
            resume_state_cas_uri: resume_v2_uri,
            committed_unix_nanos: ts_unix_nanos,
        };
        write_committed_head_v1(&run_dir, &head)?;

        let resume_state = RoundResumeStateV1 {
            schema_version: ROUND_RESUME_STATE_SCHEMA_V1,
            run_id,
            round_id: 0,
            checkpoint_asset_key: genesis_asset_key.clone(),
            recipe_key: genesis_recipe_key,
            content_digest: genesis_digest,
            blob_uri: genesis_uri,
            blob_bytes: genesis_bytes,
            parameter_count: initial_model.num_parameters(),
            committed_unix_nanos: ts_unix_nanos,
        };
        write_round_resume_state(&run_dir, &resume_state)?;

        let replay = Self::build_replay_protection(&config, 0);

        Ok(Self {
            run_id,
            swarm_instance_id,
            coordinator_peer_id,
            coordinator_generation: 0,
            config,
            current_round: 1,
            model: initial_model,
            parent_digest: genesis_digest,
            parent_recipe_key: genesis_recipe_key,
            parent_asset_key: genesis_asset_key,
            historical_digests,
            replay,
            admitted: BTreeMap::new(),
            equivocated: BTreeSet::new(),
            cclip_center: None,
            bucketing: None,
            last_bucketing_report: None,
            dp_accountant: None,
            fedasync_policy: None,
            fedasync_hinge_threshold: 0,
            fedasync_clip_norm: None,
            historical_anchors,
            manifest_signer: None,
            #[cfg(test)]
            fail_point: None,
        })
    }

    /// Restore only verified checkpoint weights onto a clean training lineage (`RestoreModeV1::WeightsOnlyRestore`, `ST-04`).
    ///
    /// Resets `cclip_center = None`, clears `fedasync_policy` and `bucketing`, keeps only the
    /// restored checkpoint as the historical anchor for `round_id`, and forces fresh session
    /// handshake onboarding with a monotonic timestamp floor.
    pub fn restore_weights_only(
        config: RoundCoordinatorConfig,
        model_template: M,
        session: &DataOpsSession,
        trust_policy: &TrustPolicy,
    ) -> Result<Self, RoundError> {
        config.validate()?;
        let run_dir = session.sink().bundle().run_dir();
        let (_head, _manifest, _status, resume_v2) = session
            .sink()
            .bundle()
            .verify_committed_head(trust_policy)?;
        if resume_v2.run_id != session.sink().bundle().run_id() {
            return Err(RoundError::RunIdMismatch {
                expected: session.sink().bundle().run_id(),
                got: resume_v2.run_id,
            });
        }
        let model: M = get_verified_checkpoint_model(
            run_dir,
            model_template,
            &resume_v2.model_digest,
            resume_v2.parameter_count,
        )?;
        let replay = Self::build_replay_protection(&config, resume_v2.replay_timestamp_floor_secs);

        let mut historical_digests = BTreeMap::new();
        historical_digests.insert(resume_v2.round_id, resume_v2.model_digest);
        let mut historical_anchors = BTreeMap::new();
        historical_anchors.insert(resume_v2.round_id, model.parameters().to_vec());

        Ok(Self {
            run_id: resume_v2.run_id,
            swarm_instance_id: resume_v2.swarm_instance_id,
            coordinator_peer_id: resume_v2.coordinator_peer_id,
            coordinator_generation: resume_v2.generation,
            config,
            current_round: resume_v2.next_round_id,
            model,
            parent_digest: resume_v2.model_digest,
            parent_recipe_key: resume_v2.recipe_key,
            parent_asset_key: resume_v2.checkpoint_asset_key,
            historical_digests,
            replay,
            admitted: BTreeMap::new(),
            equivocated: BTreeSet::new(),
            cclip_center: None,
            bucketing: None,
            last_bucketing_report: None,
            dp_accountant: None,
            fedasync_policy: None,
            fedasync_hinge_threshold: 0,
            fedasync_clip_norm: None,
            historical_anchors,
            manifest_signer: None,
            #[cfg(test)]
            fail_point: None,
        })
    }

    /// Perform full authenticated resume of model weights, optimizer state, `cclip_center`,
    /// bounded `historical_anchors`, generation counter, and replay floor from
    /// `checkpoints/committed_head.json` (`RestoreModeV1::ExactTrainingResume`, `ST-04`).
    ///
    /// Fails closed if `committed_head.json`, manifest signature/integrity, `CoordinatorResumeStateV2`,
    /// `model`, `optimizer_state` (mandatory for `UpdateKindV1::GradientV1`), `cclip_center`
    /// (mandatory for `CenteredClipping` with `round_id >= 1`), `historical_anchors`, or
    /// `config` policy digest is missing, mismatched, or corrupted. Does not mutate `optimizer`
    /// unless every verification check succeeds.
    pub fn resume_exact_training_state<O: SwarmOptimizer<Model = M>>(
        config: RoundCoordinatorConfig,
        model_template: M,
        optimizer: Option<&mut O>,
        session: &DataOpsSession,
        trust_policy: &TrustPolicy,
    ) -> Result<Self, RoundError> {
        config.validate()?;
        let run_dir = session.sink().bundle().run_dir();
        let (_head, _manifest, _status, resume_v2) = session
            .sink()
            .bundle()
            .verify_committed_head(trust_policy)?;
        if resume_v2.run_id != session.sink().bundle().run_id() {
            return Err(RoundError::RunIdMismatch {
                expected: session.sink().bundle().run_id(),
                got: resume_v2.run_id,
            });
        }

        let model: M = get_verified_checkpoint_model(
            run_dir,
            model_template,
            &resume_v2.model_digest,
            resume_v2.parameter_count,
        )?;
        let expected_cfg_hex =
            compute_base_config_policy_digest_hex(&config, model.num_parameters());
        if expected_cfg_hex != resume_v2.config_policy_digest_hex {
            return Err(RoundError::ResumeStateMismatch(format!(
                "coordinator config policy digest {} does not match committed checkpoint policy {}",
                expected_cfg_hex, resume_v2.config_policy_digest_hex
            )));
        }

        // Load and verify optimizer snapshot BEFORE mutating caller's `optimizer`!
        let loaded_opt_snapshot = if matches!(config.update_kind, UpdateKindV1::GradientV1) {
            if optimizer.is_none() {
                return Err(RoundError::ResumeStateMismatch(
                    "ExactTrainingResume with UpdateKindV1::GradientV1 requires Some(optimizer)"
                        .to_string(),
                ));
            }
            if resume_v2.round_id >= 1 {
                let opt_digest = resume_v2.optimizer_state_digest.as_ref().ok_or_else(|| {
                    RoundError::ResumeStateMismatch(
                        "missing optimizer_state_digest in committed GradientV1 checkpoint"
                            .to_string(),
                    )
                })?;
                Some(get_verified_optimizer_state_snapshot_v1(
                    run_dir,
                    opt_digest,
                    resume_v2.optimizer_state_bytes,
                )?)
            } else {
                None
            }
        } else if optimizer.is_some() {
            match resume_v2.optimizer_state_digest.as_ref() {
                Some(opt_digest) => Some(get_verified_optimizer_state_snapshot_v1(
                    run_dir,
                    opt_digest,
                    resume_v2.optimizer_state_bytes,
                )?),
                None => None,
            }
        } else {
            None
        };

        // Load and verify `cclip_center` (mandatory for `CenteredClipping` once >= 1 round committed).
        let restored_cclip_center = if matches!(
            config.aggregation,
            RobustAggregation::CenteredClipping { .. }
        ) && resume_v2.round_id >= 1
        {
            let cclip_digest = resume_v2.cclip_center_digest.as_ref().ok_or_else(|| {
                RoundError::ResumeStateMismatch(
                    "missing cclip_center_digest for CenteredClipping ExactTrainingResume"
                        .to_string(),
                )
            })?;
            Some(get_verified_cclip_center_v1(
                run_dir,
                cclip_digest,
                resume_v2.cclip_center_bytes,
                model.num_parameters(),
            )?)
        } else if let Some(cclip_digest) = resume_v2.cclip_center_digest.as_ref() {
            Some(get_verified_cclip_center_v1(
                run_dir,
                cclip_digest,
                resume_v2.cclip_center_bytes,
                model.num_parameters(),
            )?)
        } else {
            None
        };

        // Load and verify bounded `historical_anchors`.
        let restored_anchors = get_verified_historical_anchors_v1(
            run_dir,
            &resume_v2.historical_anchors_digest,
            Some(resume_v2.historical_anchors_bytes),
            model.num_parameters(),
        )?;
        if !restored_anchors.contains_key(&resume_v2.round_id) {
            return Err(RoundError::ResumeStateMismatch(format!(
                "historical_anchors missing anchor for committed round {}",
                resume_v2.round_id
            )));
        }

        // Apply optimizer restore ONLY after all other CAS/manifest/config checks succeed.
        if let (Some(opt), Some(ref snapshot)) = (optimizer, loaded_opt_snapshot) {
            opt.restore_state(snapshot)
                .map_err(|e| RoundError::OptimizerStateError(format!("{e:?}")))?;
        }

        let replay = Self::build_replay_protection(&config, resume_v2.replay_timestamp_floor_secs);

        Ok(Self {
            run_id: resume_v2.run_id,
            swarm_instance_id: resume_v2.swarm_instance_id,
            coordinator_peer_id: resume_v2.coordinator_peer_id,
            coordinator_generation: resume_v2.generation,
            config,
            current_round: resume_v2.next_round_id,
            model,
            parent_digest: resume_v2.model_digest,
            parent_recipe_key: resume_v2.recipe_key,
            parent_asset_key: resume_v2.checkpoint_asset_key,
            historical_digests: resume_v2.historical_digests,
            replay,
            admitted: BTreeMap::new(),
            equivocated: BTreeSet::new(),
            cclip_center: restored_cclip_center,
            bucketing: resume_v2.bucketing,
            last_bucketing_report: None,
            dp_accountant: None,
            fedasync_policy: resume_v2.fedasync_policy,
            fedasync_hinge_threshold: resume_v2.fedasync_hinge_threshold,
            fedasync_clip_norm: resume_v2.fedasync_clip_norm,
            historical_anchors: restored_anchors,
            manifest_signer: None,
            #[cfg(test)]
            fail_point: None,
        })
    }

    /// Dispatch coordinator restoration via an explicit [`RestoreModeV1`] (`ST-04`).
    pub fn restore_with_mode<O: SwarmOptimizer<Model = M>>(
        mode: RestoreModeV1,
        config: RoundCoordinatorConfig,
        model_template: M,
        optimizer: Option<&mut O>,
        session: &DataOpsSession,
        trust_policy: &TrustPolicy,
    ) -> Result<Self, RoundError> {
        match mode {
            RestoreModeV1::WeightsOnlyRestore => {
                Self::restore_weights_only(config, model_template, session, trust_policy)
            }
            RestoreModeV1::ExactTrainingResume => Self::resume_exact_training_state(
                config,
                model_template,
                optimizer,
                session,
                trust_policy,
            ),
        }
    }

    /// Legacy v1 coordinator resume helper (`ADR-0023` §2.2).
    ///
    /// Prefer [`Self::resume_exact_training_state`] or [`Self::restore_weights_only`] which
    /// verify `checkpoints/committed_head.json`, optimizer state, `cclip_center`, and bounded
    /// `historical_anchors` fail-closed (`ST-04`).
    #[deprecated(
        note = "use resume_exact_training_state or restore_weights_only for authenticated transactional state recovery (ST-04)"
    )]
    pub fn from_resume_state(
        config: RoundCoordinatorConfig,
        model_template: M,
        resume_state: &RoundResumeStateV1,
        session: &DataOpsSession,
    ) -> Result<Self, RoundError> {
        config.validate()?;
        if matches!(config.update_kind, UpdateKindV1::GradientV1) {
            return Err(RoundError::ResumeStateMismatch(
                "from_resume_state cannot restore GradientV1 optimizer state; use resume_exact_training_state or restore_weights_only"
                    .to_string(),
            ));
        }
        if matches!(
            config.aggregation,
            RobustAggregation::CenteredClipping { .. }
        ) && resume_state.round_id >= 1
        {
            return Err(RoundError::ResumeStateMismatch(
                "from_resume_state cannot restore CenteredClipping center; use resume_exact_training_state or restore_weights_only"
                    .to_string(),
            ));
        }
        let next_round =
            resume_state
                .round_id
                .checked_add(1)
                .ok_or(RoundError::RoundIdOverflow {
                    round_id: resume_state.round_id,
                })?;
        let run_dir = session.sink().bundle().run_dir();
        if resume_state.run_id != session.sink().bundle().run_id() {
            return Err(RoundError::RunIdMismatch {
                expected: session.sink().bundle().run_id(),
                got: resume_state.run_id,
            });
        }
        let model: M = get_verified_checkpoint_model(
            run_dir,
            model_template,
            &resume_state.content_digest,
            resume_state.parameter_count,
        )?;
        let floor_secs =
            (resume_state.committed_unix_nanos / 1_000_000_000).min(u32::MAX as u64) as u32;
        let replay = Self::build_replay_protection(&config, floor_secs);

        let mut historical_digests = BTreeMap::new();
        historical_digests.insert(resume_state.round_id, resume_state.content_digest);
        let mut historical_anchors = BTreeMap::new();
        historical_anchors.insert(resume_state.round_id, model.parameters().to_vec());
        let swarm_instance_id = default_swarm_instance_id(resume_state.run_id);
        let coordinator_peer_id = default_coordinator_peer_id(resume_state.run_id);

        Ok(Self {
            run_id: resume_state.run_id,
            swarm_instance_id,
            coordinator_peer_id,
            coordinator_generation: 0,
            config,
            current_round: next_round,
            model,
            parent_digest: resume_state.content_digest,
            parent_recipe_key: resume_state.recipe_key,
            parent_asset_key: resume_state.checkpoint_asset_key.clone(),
            historical_digests,
            replay,
            admitted: BTreeMap::new(),
            equivocated: BTreeSet::new(),
            cclip_center: None,
            bucketing: None,
            last_bucketing_report: None,
            dp_accountant: None,
            fedasync_policy: None,
            fedasync_hinge_threshold: 0,
            fedasync_clip_norm: None,
            historical_anchors,
            manifest_signer: None,
            #[cfg(test)]
            fail_point: None,
        })
    }

    /// Attach an Ed25519 manifest signing key so Step 3 signs every committed round manifest
    /// prior to publishing `checkpoints/committed_head.json` (`ST-04`).
    pub fn with_manifest_signer(mut self, signer: KeyPair) -> Self {
        self.manifest_signer = Some(signer);
        self
    }

    /// Configure or clear the Step 3 manifest signing key (`ST-04`).
    pub fn set_manifest_signer(&mut self, signer: Option<KeyPair>) {
        self.manifest_signer = signer;
    }

    #[cfg(test)]
    pub(crate) fn set_test_commit_fail_point(&mut self, fp: CommitFailPoint) {
        self.fail_point = Some(fp);
    }

    /// Configure conservative `$s$-Bucketing` pre-aggregation for subsequent rounds (`ADR-0019` §4, `W11-01`).
    pub fn set_bucketing(&mut self, bucketing: Option<BucketingConfig>) {
        self.bucketing = bucketing;
        self.coordinator_generation = self.coordinator_generation.saturating_add(1);
    }

    /// Override the coordinator's [`swarm_instance_id`](Self::swarm_instance_id) and
    /// [`coordinator_peer_id`](Self::coordinator_peer_id), bumping `coordinator_generation`.
    pub fn with_coordinator_identity(
        mut self,
        swarm_instance_id: [u8; 16],
        coordinator_peer_id: PeerId,
    ) -> Self {
        self.swarm_instance_id = swarm_instance_id;
        self.coordinator_peer_id = coordinator_peer_id;
        self.coordinator_generation = self.coordinator_generation.saturating_add(1);
        self
    }

    /// Borrow the `$s$-Bucketing` execution report from the most recently finalized round.
    pub const fn last_bucketing_report(&self) -> Option<&BucketExecutionReport> {
        self.last_bucketing_report.as_ref()
    }

    /// Borrow the current [`CenteredClipping`] reference center $v^{(r-1)}$, if initialized.
    pub fn cclip_center(&self) -> Option<&[f32]> {
        self.cclip_center.as_deref()
    }

    /// Explicitly seed or restore the [`CenteredClipping`] reference center $v^{(r-1)}$ (`ADR-0019` §4).
    pub fn set_cclip_center(&mut self, center: Vec<f32>) -> Result<(), RoundError> {
        let expected = self.model.num_parameters();
        if center.len() != expected {
            return Err(RoundError::DimensionMismatch {
                expected,
                got: center.len(),
            });
        }
        for &val in &center {
            if val.is_nan() {
                return Err(RoundError::Aggregation(CoreError::NanDetected));
            }
            if val.is_infinite() {
                return Err(RoundError::Aggregation(CoreError::InfDetected));
            }
        }
        self.cclip_center = Some(center);
        Ok(())
    }

    /// Bound run identifier.
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Scoped swarm instance identifier expected on v2 deliveries and session handshakes.
    pub const fn swarm_instance_id(&self) -> [u8; 16] {
        self.swarm_instance_id
    }

    /// Local coordinator recipient [`PeerId`] expected as `intended_recipient` on v2 deliveries.
    pub const fn coordinator_peer_id(&self) -> PeerId {
        self.coordinator_peer_id
    }

    /// Monotonic coordinator generation incremented on every committed round or policy mutation.
    pub const fn coordinator_generation(&self) -> u64 {
        self.coordinator_generation
    }

    /// Number of active entries currently tracked by `ReplayProtection`.
    pub fn replay_cache_size(&self) -> usize {
        self.replay.cache_size()
    }

    /// Number of registered, pending v2 session recovery challenges.
    pub fn replay_pending_challenge_count(&self) -> usize {
        self.replay.pending_challenge_count()
    }

    /// Number of peers revoked for equivocation in `current_round`.
    pub fn equivocated_count(&self) -> usize {
        self.equivocated.len()
    }

    /// Deterministic SHA-256 context/aggregation policy digest bound into v2 updates & handshakes.
    pub fn context_policy_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(CONTEXT_POLICY_DIGEST_DOMAIN_V1);
        hasher.update(self.swarm_instance_id);
        hasher.update(self.run_id.as_bytes());
        hasher.update(self.coordinator_peer_id.as_bytes());
        hasher.update((self.model.num_parameters() as u64).to_le_bytes());

        hash_round_coordinator_config(&mut hasher, &self.config);

        match self.fedasync_policy {
            Some(p) => {
                hasher.update([1u8]);
                hasher.update(p.base_alpha.to_bits().to_le_bytes());
                hasher.update(p.max_staleness_rounds.to_le_bytes());
                hasher.update(p.polynomial_decay_exponent.to_bits().to_le_bytes());
                hasher.update(self.fedasync_hinge_threshold.to_le_bytes());
                match self.fedasync_clip_norm {
                    Some(tau) => {
                        hasher.update([1u8]);
                        hasher.update(tau.to_bits().to_le_bytes());
                    }
                    None => hasher.update([0u8]),
                }
            }
            None => hasher.update([0u8]),
        }

        match self.bucketing {
            Some(b) => {
                hasher.update([1u8]);
                hasher.update((b.bucket_size as u64).to_le_bytes());
                hasher.update((b.byzantine_count as u64).to_le_bytes());
                hasher.update(b.round_seed);
                let rem_tag: u8 = match b.remainder_policy {
                    BucketRemainderPolicy::StrictExactMultiple => 1,
                    BucketRemainderPolicy::DropRemainderDeterministic => 2,
                };
                hasher.update([rem_tag]);
            }
            None => hasher.update([0u8]),
        }

        hasher.finalize().into()
    }

    /// Export an immutable [`ExpectedRoundContextV1`] snapshot for Stage B validation (`ST-01`).
    pub fn expected_round_context(&self) -> ExpectedRoundContextV1 {
        ExpectedRoundContextV1 {
            swarm_instance_id: self.swarm_instance_id,
            coordinator_peer_id: self.coordinator_peer_id,
            run_id: self.run_id,
            active_round_id: self.current_round,
            max_staleness: self.config.max_staleness,
            accepted_parent_digest: self.parent_digest,
            historical_digests_snapshot: self.historical_digests.clone(),
            historical_anchor_rounds: self.historical_anchors.keys().copied().collect(),
            expected_update_kind: self.config.update_kind,
            expected_param_count: self.model.num_parameters(),
            cohort_allowlist_snapshot: self.config.authorized_cohort.clone(),
            equivocated_snapshot: self.equivocated.clone(),
            context_policy_digest: self.context_policy_digest(),
            fedasync_policy: self.fedasync_policy,
            coordinator_generation: self.coordinator_generation,
        }
    }

    /// Register a receiver-issued v2 session handshake challenge (`ST-03`).
    pub fn register_session_challenge_v2(
        &mut self,
        challenge: SessionHandshakeChallengeV2,
    ) -> Result<(), RoundError> {
        if challenge.swarm_instance_id != self.swarm_instance_id {
            return Err(RoundError::SwarmInstanceMismatch {
                expected: self.swarm_instance_id,
                got: challenge.swarm_instance_id,
            });
        }
        let challenge_run_id = RunId::from_bytes(challenge.run_id);
        if challenge_run_id != self.run_id {
            return Err(RoundError::RunIdMismatch {
                expected: self.run_id,
                got: challenge_run_id,
            });
        }
        if challenge.receiver_peer != self.coordinator_peer_id {
            return Err(RoundError::UnauthorizedPeer(challenge.receiver_peer));
        }
        let expected_policy = self.context_policy_digest();
        if challenge.policy_digest != expected_policy {
            return Err(RoundError::ContextPolicyDigestMismatch {
                expected: expected_policy,
                got: challenge.policy_digest,
            });
        }
        self.replay
            .register_recovery_challenge_v2(challenge)
            .map_err(RoundError::ReplayRejected)
    }

    /// Complete a receiver-bound v2 session handshake atomically (`ST-03`).
    pub fn complete_session_handshake_v2(
        &mut self,
        sender_public_key: &[u8; 32],
        signature: &Signature,
        current_time: u32,
    ) -> Result<ReplayRecordKeyV2, RoundError> {
        self.replay
            .complete_recovery_handshake_v2(
                self.swarm_instance_id,
                *self.run_id.as_bytes(),
                sender_public_key,
                signature,
                current_time,
            )
            .map_err(RoundError::ReplayRejected)
    }

    /// Current active round number (starts at `1` after genesis initialization).
    pub const fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Reference to the current committed global model ($w_{\text{parent}}$).
    pub const fn model(&self) -> &M {
        &self.model
    }

    /// Content digest of the current committed global model checkpoint.
    pub const fn parent_digest(&self) -> CheckpointContentDigestV1 {
        self.parent_digest
    }

    /// Recipe key of the current committed global model checkpoint.
    pub const fn parent_recipe_key(&self) -> CheckpointRecipeKeyV1 {
        self.parent_recipe_key
    }

    /// Number of currently admitted non-equivocating peers for `current_round`.
    pub fn admitted_count(&self) -> usize {
        self.admitted.len()
    }

    /// Whether `admitted_count() >= config.min_quorum`.
    pub fn has_quorum(&self) -> bool {
        self.admitted.len() >= self.config.min_quorum
    }

    /// Legacy unverified round submission entrypoint — rejected fail-closed (`ST-01`).
    ///
    /// Use [`Self::prepare_verified_delivery`] -> [`ExpectedRoundContextV1::validate_sync_submission`]
    /// -> [`Self::commit_prepared_submission`] -> [`Self::submit_verified_update`] (or the
    /// convenience [`Self::submit_verified_delivery`]).
    #[deprecated(
        note = "unverified ingress fails closed; use submit_verified_delivery or the 3-stage pipeline"
    )]
    #[allow(deprecated)]
    pub fn submit_peer_update(
        &mut self,
        _submission: PeerRoundSubmissionV1,
        _now_secs: u64,
    ) -> Result<SubmissionDecision, RoundError> {
        Err(RoundError::UnverifiedSubmissionRejected)
    }

    /// Stage A (`&self`, zero state mutation): bounded-deserialize and cryptographically verify a
    /// signed [`BundledGossipDeliveryV2`] wire frame against `&self.replay` without consuming any
    /// replay sequence state (`ST-01`, `ST-03`).
    pub fn prepare_verified_delivery(
        &self,
        delivery_bytes: &[u8],
        max_delivery_bytes: usize,
        now_secs: u64,
    ) -> Result<PreparedSubmissionV1, RoundError> {
        let delivery =
            BundledGossipDeliveryV2::from_bytes_bounded(delivery_bytes, max_delivery_bytes)?;
        self.prepare_verified_delivery_struct(&delivery, now_secs)
    }

    /// Stage A (`&self`, zero state mutation) over an already-deserialized
    /// [`BundledGossipDeliveryV2`] (`ST-01`, `ST-03`).
    pub fn prepare_verified_delivery_struct(
        &self,
        delivery: &BundledGossipDeliveryV2,
        now_secs: u64,
    ) -> Result<PreparedSubmissionV1, RoundError> {
        let now_u32 = u32::try_from(now_secs).map_err(|_| {
            RoundError::ReplayRejected(ReplayError::Expired {
                ts: delivery.wrapper.timestamp_secs,
                now: u32::MAX,
                window: self.config.max_clock_skew_secs,
            })
        })?;
        let (originator, replay_admission) = delivery.verify_for_recipient_v2_prepared(
            self.coordinator_peer_id,
            &self.replay,
            now_u32,
        )?;
        let payload = decode_prepared_payload_v2(&delivery.inner, originator)?;
        Ok(PreparedSubmissionV1 {
            sender: originator,
            swarm_instance_id: delivery.inner.swarm_instance_id,
            run_id: RunId::from_bytes(delivery.inner.run_id),
            round_id: delivery.inner.round_id,
            base_checkpoint_digest: CheckpointContentDigestV1(
                delivery.inner.base_checkpoint_digest,
            ),
            context_policy_digest: delivery.inner.context_policy_digest,
            payload_kind: delivery.inner.payload_kind,
            reported_examples: if delivery.inner.reported_examples == 0 {
                None
            } else {
                Some(delivery.inner.reported_examples)
            },
            payload,
            replay_admission,
        })
    }

    /// Stage C (synchronous round): verify `coordinator_generation` still matches, commit the
    /// prepared replay sequence atomically, and emit a sealed [`VerifiedRoundSubmissionV1`] (`ST-01`).
    pub fn commit_prepared_submission(
        &mut self,
        validated: ValidatedRoundSubmissionV1,
    ) -> Result<VerifiedRoundSubmissionV1, RoundError> {
        if validated.coordinator_generation != self.coordinator_generation {
            return Err(RoundError::StaleCoordinatorGeneration {
                expected: self.coordinator_generation,
                got: validated.coordinator_generation,
            });
        }
        let sequence = validated.replay_admission.sequence();
        self.replay
            .commit_prepared_admission(validated.replay_admission)
            .map_err(RoundError::ReplayRejected)?;
        Ok(VerifiedRoundSubmissionV1 {
            coordinator_generation: validated.coordinator_generation,
            round_id: validated.target_round_id,
            sender: validated.sender,
            sequence,
            canonical_payload_digest: validated.canonical_payload_digest,
            values: validated.values,
            reported_examples: validated.reported_examples,
        })
    }

    /// Admit a sealed [`VerifiedRoundSubmissionV1`] produced by Stage C (`ST-01`).
    pub fn submit_verified_update(
        &mut self,
        verified: VerifiedRoundSubmissionV1,
    ) -> Result<SubmissionDecision, RoundError> {
        if verified.coordinator_generation != self.coordinator_generation {
            return Err(RoundError::StaleCoordinatorGeneration {
                expected: self.coordinator_generation,
                got: verified.coordinator_generation,
            });
        }
        if verified.round_id != self.current_round {
            return Err(RoundError::StaleRoundRejected {
                submission_round: verified.round_id,
                current_round: self.current_round,
                max_staleness: self.config.max_staleness,
            });
        }
        if self.equivocated.contains(&verified.sender) {
            return Err(RoundError::EquivocationRejected {
                sender: verified.sender,
                round_id: self.current_round,
            });
        }

        if let Some(existing) = self.admitted.get(&verified.sender) {
            if existing.payload_digest == verified.canonical_payload_digest {
                return Ok(SubmissionDecision::DuplicateIdempotentAck);
            }
            self.admitted.remove(&verified.sender);
            self.equivocated.insert(verified.sender);
            return Err(RoundError::EquivocationRejected {
                sender: verified.sender,
                round_id: self.current_round,
            });
        }

        let grad_update = GradientUpdate {
            sender: *verified.sender.as_bytes(),
            sequence: verified.sequence,
            gradients: verified.values,
            round_id: self.current_round,
        };
        self.admitted.insert(
            verified.sender,
            AdmittedPeerEntry {
                payload_digest: verified.canonical_payload_digest,
                update: grad_update,
                reported_examples: verified.reported_examples,
            },
        );

        Ok(SubmissionDecision::Admitted {
            admitted_count: self.admitted.len(),
            min_quorum: self.config.min_quorum,
        })
    }

    /// Execute Stages A -> B -> C + admission atomically for a synchronous signed wire frame (`ST-01`).
    pub fn submit_verified_delivery(
        &mut self,
        delivery_bytes: &[u8],
        max_delivery_bytes: usize,
        now_secs: u64,
    ) -> Result<SubmissionDecision, RoundError> {
        let prepared =
            self.prepare_verified_delivery(delivery_bytes, max_delivery_bytes, now_secs)?;
        let ctx = self.expected_round_context();
        let validated = ctx.validate_sync_submission(prepared)?;
        let verified = self.commit_prepared_submission(validated)?;
        self.submit_verified_update(verified)
    }

    /// Finalize the current round, applying `transforms`, running the configured
    /// [`RobustAggregation`] / [`WeightingPolicyV1`], updating the global model, evaluating
    /// optional `fitness`, and committing the new checkpoint to the physical CAS store and
    /// Finalize the active round cleanly using transactional 5-step commit (`ADR-0023` §1, §2.2, `ST-04`):
    ///
    /// 1. Validate quorum, bounds, transforms, aggregation, candidate optimizer step, and fitness
    ///    strictly in memory without mutating `self`, `*server_optimizer`, or `session`.
    /// 2. Pre-debit DP ledger if configured and write immutable CAS blobs (`model`, optional
    ///    `optimizer_state`, optional `cclip_center`, `historical_anchors`, `CoordinatorResumeStateV2`),
    ///    followed by DataOps materialization & telemetry.
    /// 3. Finalize (and optionally sign) `manifest.json` + archived generation manifest.
    /// 4. Atomically swap & sync `checkpoints/committed_head.json` (authoritative commit point).
    /// 5. Commit staged state into memory (`self` + `*server_optimizer`) infallibly.
    pub fn finalize_round<O, F>(
        &mut self,
        transforms: &[&dyn UpdateTransform],
        server_optimizer: Option<&mut O>,
        fitness: Option<&F>,
        session: &mut DataOpsSession,
        ts_unix_nanos: u64,
    ) -> Result<RoundCommitSummaryV1, RoundError>
    where
        O: SwarmOptimizer<Model = M> + Clone,
        F: FitnessFunction<Model = M>,
    {
        // Step 1: Pure in-memory preparation & validation (`ST-04`).
        if self.admitted.len() < self.config.min_quorum {
            return Err(RoundError::InsufficientQuorum {
                admitted: self.admitted.len(),
                min_quorum: self.config.min_quorum,
            });
        }
        let next_round = self
            .current_round
            .checked_add(1)
            .ok_or(RoundError::RoundIdOverflow {
                round_id: self.current_round,
            })?;
        let next_generation = self.coordinator_generation.checked_add(1).ok_or(
            RoundError::CoordinatorGenerationOverflow {
                generation: self.coordinator_generation,
            },
        )?;

        let mut admitted_peers = Vec::with_capacity(self.admitted.len());
        let mut frozen_raw_inputs = BTreeMap::new();
        let mut working_updates = Vec::with_capacity(self.admitted.len());
        let mut weight_mismatch_warnings = Vec::new();

        for (&peer_id, entry) in &self.admitted {
            admitted_peers.push(peer_id);
            frozen_raw_inputs.insert(peer_id, entry.update.gradients.clone());
            working_updates.push(entry.update.clone());

            if let WeightingPolicyV1::RegistryCertified {
                ref peer_weights, ..
            } = self.config.weighting_policy
            {
                if let (Some(reported), Some(&certified)) =
                    (entry.reported_examples, peer_weights.get(&peer_id))
                {
                    if reported != certified {
                        weight_mismatch_warnings.push(format!(
                            "peer {:?} reported_examples={} differs from certified weight={}; using certified weight",
                            peer_id, reported, certified
                        ));
                    }
                }
            }
        }

        let mut staged_transform_audits = Vec::with_capacity(transforms.len());
        for &transform in transforms {
            struct DynTransformRef<'a>(&'a dyn UpdateTransform);
            impl UpdateTransform for DynTransformRef<'_> {
                fn transform(&self, update: GradientUpdate) -> GradientUpdate {
                    self.0.transform(update)
                }
                fn name(&self) -> &str {
                    self.0.name()
                }
                fn is_core_trusted(&self) -> bool {
                    self.0.is_core_trusted()
                }
            }
            let wrapper = DynTransformRef(transform);
            let (next_updates, mut audit): (Vec<GradientUpdate>, TransformAuditV0) =
                apply_update_transforms(working_updates, &wrapper);
            audit.round_id = self.current_round;
            staged_transform_audits.push(audit);
            working_updates = next_updates;
        }

        let (aggregated_vec, bucketing_report) =
            self.aggregate_working_updates(&admitted_peers, &working_updates)?;

        let mut candidate_model = self.model.clone();
        let mut staged_optimizer: Option<O> = None;
        match self.config.update_kind {
            UpdateKindV1::ModelDeltaV1 {
                server_learning_rate,
            } => {
                let lr_f64 = server_learning_rate as f64;
                let parent_params = self.model.parameters();
                let mut new_params = Vec::with_capacity(parent_params.len());
                for (&w_parent, &delta) in parent_params.iter().zip(aggregated_vec.iter()) {
                    let updated = (w_parent as f64) + lr_f64 * (delta as f64);
                    let updated_f32 = updated as f32;
                    if !updated_f32.is_finite() {
                        return Err(RoundError::Aggregation(CoreError::InfDetected));
                    }
                    new_params.push(updated_f32);
                }
                candidate_model
                    .load_parameters(&new_params)
                    .map_err(RoundError::Aggregation)?;
            }
            UpdateKindV1::GradientV1 => {
                let opt = server_optimizer.as_deref().ok_or_else(|| {
                    RoundError::OptimizerStep(
                        "UpdateKindV1::GradientV1 requires a server SwarmOptimizer".to_string(),
                    )
                })?;
                let mut candidate_opt = opt.clone();
                candidate_opt
                    .step(&mut candidate_model, &aggregated_vec)
                    .map_err(|e| RoundError::OptimizerStep(format!("{e:?}")))?;
                if candidate_model.parameters().iter().any(|p| !p.is_finite()) {
                    return Err(RoundError::Aggregation(CoreError::NanDetected));
                }
                staged_optimizer = Some(candidate_opt);
            }
        }

        let optimizer_snapshot = match (&staged_optimizer, server_optimizer.as_deref()) {
            (Some(cand), _) => cand
                .snapshot_state()
                .map_err(|e| RoundError::OptimizerStateError(format!("{e:?}")))?,
            (None, Some(opt)) => opt
                .snapshot_state()
                .map_err(|e| RoundError::OptimizerStateError(format!("{e:?}")))?,
            (None, None) => None,
        };

        let fitness_score = match fitness {
            Some(evaluator) => {
                let score = evaluator
                    .evaluate(&candidate_model)
                    .map_err(|e| RoundError::FitnessEvaluation(format!("{e:?}")))?;
                if !score.is_finite() {
                    return Err(RoundError::FitnessEvaluation(
                        "fitness evaluation returned non-finite score".to_string(),
                    ));
                }
                Some(score)
            }
            None => None,
        };

        let committed_round = self.current_round;
        let next_historical_anchors = self.compute_next_historical_anchors(
            committed_round,
            candidate_model.parameters().to_vec(),
        );
        let equivocated_peers: Vec<PeerId> = self.equivocated.iter().copied().collect();
        let telemetry_admitted_count = admitted_peers.len();
        let telemetry_equivocated_count = self.equivocated.len();
        let checkpoint_asset_key = format!("checkpoint://round/{committed_round}");
        let node_key = format!("train/round_{committed_round}");

        let plan = StagedRoundCommit {
            committed_round,
            next_round,
            next_generation,
            candidate_model,
            optimizer_snapshot,
            next_cclip_center: Some(aggregated_vec),
            next_historical_anchors,
            next_historical_digests_without_new: self.historical_digests.clone(),
            next_bucketing_report: bucketing_report,
            checkpoint_asset_key,
            node_key,
            op_type: "swarm_round_aggregate".to_string(),
            output_rows: Some(admitted_peers.len() as u64),
            admitted_peers,
            equivocated_peers,
            frozen_raw_inputs,
            weight_mismatch_warnings,
            fitness_score,
            staged_transform_audits,
            dp_subsampling_q_override: None,
            dp_explicit_tau: None,
            emit_standard_telemetry: true,
            clear_round_admitted: true,
            ts_unix_nanos,
            telemetry_admitted_count,
            telemetry_equivocated_count,
        };

        let summary = self.commit_round_plan_v1(plan, session)?;
        if let (Some(opt), Some(committed_opt)) = (server_optimizer, staged_optimizer) {
            *opt = committed_opt;
        }
        Ok(summary)
    }

    #[cfg(test)]
    fn check_test_fail_point(&mut self, expected: CommitFailPoint) -> Result<(), RoundError> {
        if self.fail_point == Some(expected) {
            self.fail_point = None;
            return Err(RoundError::Io(io::Error::other(format!(
                "injected test fault at {expected:?}"
            ))));
        }
        Ok(())
    }

    fn commit_round_plan_v1(
        &mut self,
        plan: StagedRoundCommit<M>,
        session: &mut DataOpsSession,
    ) -> Result<RoundCommitSummaryV1, RoundError> {
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        // Step 2a: Pre-release DP debit if enabled (permanently recorded once debited).
        self.pre_release_dp_debit_if_enabled(
            session,
            plan.committed_round,
            plan.ts_unix_nanos,
            plan.dp_explicit_tau,
            plan.dp_subsampling_q_override,
        )?;

        // Step 2b: Write all immutable CAS dependencies via write-once CAS blobs.
        let (content_digest, blob_uri, blob_bytes) =
            put_checkpoint_model(&run_dir, &plan.candidate_model)?;

        let (optimizer_state_digest, optimizer_state_cas_uri, optimizer_state_bytes) =
            match plan.optimizer_snapshot.as_ref() {
                Some(snap) => {
                    let (d, u, b) = put_optimizer_state_snapshot_v1(&run_dir, snap)?;
                    (Some(d), Some(u), Some(b))
                }
                None => (None, None, None),
            };
        let (cclip_center_digest, cclip_center_cas_uri, cclip_center_bytes) =
            match plan.next_cclip_center.as_deref() {
                Some(center) => {
                    let (d, u, b) = put_cclip_center_v1(&run_dir, center)?;
                    (Some(d), Some(u), Some(b))
                }
                None => (None, None, None),
            };
        let (historical_anchors_digest, historical_anchors_cas_uri, historical_anchors_bytes) =
            put_historical_anchors_v1(&run_dir, &plan.next_historical_anchors)?;

        let round_node = NodeV1 {
            node_key: plan.node_key.clone(),
            node_id: Some(node_id_from_key(&plan.node_key)),
            op_kind: OpKind::Train,
            op_type: plan.op_type,
            inputs: vec![AssetRefV1 {
                asset_key: self.parent_asset_key.clone(),
                fingerprint: Some(self.parent_digest.to_hex()),
            }],
            outputs: vec![AssetRefV1 {
                asset_key: plan.checkpoint_asset_key.clone(),
                fingerprint: Some(content_digest.to_hex()),
            }],
            params: CanonParams::new(),
            code_ref: Some(format!("swarm-torch@{}", env!("CARGO_PKG_VERSION"))),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
            execution_hint: None,
            cache_policy: None,
            materialization_policy: None,
            resources: None,
            op_hash: None,
        };
        let node_def_digest = node_def_hash_v1(&round_node)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let recipe_key = CheckpointRecipeKeyV1::derive(
            self.run_id,
            plan.committed_round,
            &node_def_digest,
            &self.parent_recipe_key,
        );

        let mut next_historical_digests = plan.next_historical_digests_without_new;
        next_historical_digests.insert(plan.committed_round, content_digest);
        let replay_timestamp_floor_secs =
            (plan.ts_unix_nanos / 1_000_000_000).min(u32::MAX as u64) as u32;
        let param_count = plan.candidate_model.num_parameters();
        let config_policy_digest_hex =
            compute_base_config_policy_digest_hex(&self.config, param_count);

        let resume_v2 = CoordinatorResumeStateV2 {
            schema_version: COORDINATOR_RESUME_STATE_SCHEMA_V2,
            run_id: self.run_id,
            swarm_instance_id: self.swarm_instance_id,
            coordinator_peer_id: self.coordinator_peer_id,
            generation: plan.next_generation,
            round_id: plan.committed_round,
            next_round_id: plan.next_round,
            checkpoint_asset_key: plan.checkpoint_asset_key.clone(),
            recipe_key,
            model_digest: content_digest,
            model_blob_uri: blob_uri.clone(),
            model_blob_bytes: blob_bytes,
            parameter_count: param_count,
            optimizer_state_digest,
            optimizer_state_cas_uri,
            optimizer_state_bytes,
            cclip_center_digest,
            cclip_center_cas_uri,
            cclip_center_bytes,
            historical_anchors_digest,
            historical_anchors_cas_uri,
            historical_anchors_bytes,
            historical_digests: next_historical_digests.clone(),
            replay_timestamp_floor_secs,
            config_policy_digest_hex,
            fedasync_policy: self.fedasync_policy,
            fedasync_hinge_threshold: self.fedasync_hinge_threshold,
            fedasync_clip_norm: self.fedasync_clip_norm,
            bucketing: self.bucketing,
            committed_unix_nanos: plan.ts_unix_nanos,
        };
        let (_resume_v2_digest, resume_v2_uri, _resume_v2_bytes) =
            put_coordinator_resume_state_v2(&run_dir, &resume_v2)?;

        // Step 2c: Record DataOps materializations & telemetry (preserving pending_transform_audits
        // on failure so un-committed transforms do not leak into subsequent retries).
        let prev_audit_len = session.pending_transform_audit_len();
        for audit in &plan.staged_transform_audits {
            session.record_transform_applied(audit);
        }
        if let Err(err) = session.materialize_node_outputs(
            &round_node,
            &[OutputSpec {
                asset_key: plan.checkpoint_asset_key.clone(),
                schema: None,
                rows: plan.output_rows,
                bytes: Some(blob_bytes),
            }],
            plan.ts_unix_nanos,
            CacheDecisionV0::Miss,
            1,
        ) {
            session.truncate_pending_transform_audits(prev_audit_len);
            return Err(RoundError::Io(err));
        }
        session.bind_output_cas_uri(
            &plan.checkpoint_asset_key,
            &blob_uri,
            "application/x-swarmtorch-model-v1",
        )?;

        if plan.emit_standard_telemetry {
            let trace_id = TraceId::from_bytes(*self.run_id.as_bytes());
            let mut span_hasher = Sha256::new();
            span_hasher.update(b"swarmtorch.round.span.v1");
            span_hasher.update(self.run_id.as_bytes());
            span_hasher.update(plan.committed_round.to_le_bytes());
            let span_digest = span_hasher.finalize();
            let mut span_bytes = [0u8; 8];
            span_bytes.copy_from_slice(&span_digest[..8]);
            if span_bytes == [0u8; 8] {
                span_bytes[0] = 1;
            }
            let span_id = SpanId::from_bytes(span_bytes);

            let mut attrs = AttrMap::new();
            attrs.insert(
                "swarmtorch.round_id".to_string(),
                AttrValue::I64(plan.committed_round as i64),
            );
            attrs.insert(
                "swarmtorch.admitted_peers".to_string(),
                AttrValue::I64(plan.telemetry_admitted_count as i64),
            );
            attrs.insert(
                "swarmtorch.equivocated_peers".to_string(),
                AttrValue::I64(plan.telemetry_equivocated_count as i64),
            );
            attrs.insert(
                "swarmtorch.checkpoint_digest".to_string(),
                AttrValue::Str(content_digest.to_hex()),
            );

            session.sink().emit_span(&SpanRecord {
                schema_version: 1,
                trace_id,
                span_id,
                parent_span_id: None,
                name: format!("round/{}", plan.committed_round),
                start_unix_nanos: plan.ts_unix_nanos.saturating_sub(1_000_000),
                end_unix_nanos: Some(plan.ts_unix_nanos),
                attrs: attrs.clone(),
            })?;
            session.sink().emit_event(&EventRecord {
                schema_version: 1,
                ts_unix_nanos: plan.ts_unix_nanos,
                trace_id,
                span_id: Some(span_id),
                name: "coordinator/round_committed".to_string(),
                attrs: attrs.clone(),
            })?;
            if let Some(score) = plan.fitness_score {
                session.sink().emit_metric(&MetricRecord {
                    schema_version: 1,
                    ts_unix_nanos: plan.ts_unix_nanos,
                    trace_id,
                    span_id: Some(span_id),
                    name: "round_fitness".to_string(),
                    value: score as f64,
                    unit: Some("score".to_string()),
                    attrs,
                })?;
            }
        }

        #[cfg(test)]
        self.check_test_fail_point(CommitFailPoint::AfterCasBeforeManifest)?;

        // Step 3: Finalize (and optionally sign) `manifest.json` + archived generation manifest.
        let signer_ref = self.manifest_signer.as_ref();
        let (_manifest, manifest_sha256) = session
            .sink()
            .finalize_checkpoint_manifest(plan.next_generation, signer_ref)?;

        #[cfg(test)]
        self.check_test_fail_point(CommitFailPoint::AfterManifestBeforeCommittedHead)?;

        // Step 4: Authoritative Logical Commit Point — atomic swap & sync of `committed_head.json`.
        let head = CommittedHeadV1 {
            schema_version: COMMITTED_HEAD_SCHEMA_V1,
            run_id: self.run_id,
            generation: plan.next_generation,
            round_id: plan.committed_round,
            manifest_sha256,
            resume_state_cas_uri: resume_v2_uri,
            committed_unix_nanos: plan.ts_unix_nanos,
        };
        write_committed_head_v1(&run_dir, &head)?;

        let legacy_resume_state = RoundResumeStateV1 {
            schema_version: ROUND_RESUME_STATE_SCHEMA_V1,
            run_id: self.run_id,
            round_id: plan.committed_round,
            checkpoint_asset_key: plan.checkpoint_asset_key.clone(),
            recipe_key,
            content_digest,
            blob_uri: blob_uri.clone(),
            blob_bytes,
            parameter_count: param_count,
            committed_unix_nanos: plan.ts_unix_nanos,
        };
        write_round_resume_state(&run_dir, &legacy_resume_state)?;

        #[cfg(test)]
        self.check_test_fail_point(CommitFailPoint::AfterCommittedHeadBeforeMemory)?;

        // Step 5: Infallible in-memory state transition.
        self.model = plan.candidate_model;
        self.parent_digest = content_digest;
        self.parent_recipe_key = recipe_key;
        self.parent_asset_key = plan.checkpoint_asset_key.clone();
        self.historical_digests = next_historical_digests;
        self.cclip_center = plan.next_cclip_center;
        self.historical_anchors = plan.next_historical_anchors;
        self.last_bucketing_report = plan.next_bucketing_report;
        self.current_round = plan.next_round;
        self.coordinator_generation = plan.next_generation;
        if plan.clear_round_admitted {
            self.admitted.clear();
            self.equivocated.clear();
        }

        Ok(RoundCommitSummaryV1 {
            round_id: plan.committed_round,
            admitted_peers: plan.admitted_peers,
            equivocated_peers: plan.equivocated_peers,
            checkpoint_asset_key: plan.checkpoint_asset_key,
            recipe_key,
            content_digest,
            blob_uri,
            fitness_score: plan.fitness_score,
            frozen_raw_inputs: plan.frozen_raw_inputs,
            weight_mismatch_warnings: plan.weight_mismatch_warnings,
        })
    }

    fn aggregate_working_updates(
        &self,
        admitted_peers: &[PeerId],
        updates: &[GradientUpdate],
    ) -> Result<(Vec<f32>, Option<BucketExecutionReport>), RoundError> {
        if let WeightingPolicyV1::RegistryCertified {
            ref peer_weights,
            max_weight_share,
        } = self.config.weighting_policy
        {
            if !matches!(self.config.aggregation, RobustAggregation::FedAvg)
                || self.bucketing.is_some()
            {
                return Err(RoundError::Config(
                    RoundConfigError::IncompatibleWeightingPolicy,
                ));
            }
            let out = aggregate_registry_certified_fedavg(
                admitted_peers,
                updates,
                peer_weights,
                max_weight_share,
            )?;
            return Ok((out, None));
        }

        if let Some(mut b_cfg) = self.bucketing {
            // Mix round_id into the bucketing round_seed so distinct rounds do not repeat the same permutation.
            let mut seed_hasher = Sha256::new();
            seed_hasher.update(b"swarmtorch.coordinator.bucketing.seed.v1");
            seed_hasher.update(b_cfg.round_seed);
            seed_hasher.update(self.run_id.as_bytes());
            seed_hasher.update(self.current_round.to_le_bytes());
            let seed_digest = seed_hasher.finalize();
            b_cfg.round_seed.copy_from_slice(&seed_digest);

            let budget = AggregationBudget::default_pairwise();
            if matches!(self.config.aggregation, RobustAggregation::FedAvg) {
                let (bucket_updates, report) = bucket_updates_v1(updates, &b_cfg)?;
                let out = FedAvg.aggregate(&bucket_updates)?;
                return Ok((out, Some(report)));
            }

            let post_rule = match &self.config.aggregation {
                RobustAggregation::FedAvg => unreachable!(),
                RobustAggregation::Median => PostBucketingRule::CoordinateMedian,
                RobustAggregation::TrimmedMean { trim_ratio } => PostBucketingRule::TrimmedMean {
                    trim_ratio: *trim_ratio,
                },
                #[cfg(feature = "robust-aggregation")]
                RobustAggregation::Krum { .. } => PostBucketingRule::Krum,
                #[cfg(feature = "robust-aggregation")]
                RobustAggregation::MultiKrum { num_selected, .. } => {
                    PostBucketingRule::MultiKrum { m: *num_selected }
                }
                #[cfg(feature = "robust-aggregation")]
                RobustAggregation::Bulyan { .. } => PostBucketingRule::Bulyan,
                RobustAggregation::CenteredClipping {
                    tau,
                    max_iterations,
                } => PostBucketingRule::CenteredClipping {
                    tau: *tau,
                    max_iterations: *max_iterations,
                },
            };

            let (out, report) = aggregate_with_bucketing_v1(
                updates,
                &b_cfg,
                post_rule,
                self.cclip_center.as_deref(),
                &budget,
            )?;
            return Ok((out, Some(report)));
        }

        let out = match &self.config.aggregation {
            RobustAggregation::FedAvg => FedAvg.aggregate(updates),
            RobustAggregation::Median => CoordinateMedian.aggregate(updates),
            RobustAggregation::TrimmedMean { trim_ratio } => {
                TrimmedMean::try_new(*trim_ratio)?.aggregate(updates)
            }
            #[cfg(feature = "robust-aggregation")]
            RobustAggregation::Krum { num_byzantine } => {
                Krum::new(*num_byzantine).aggregate(updates)
            }
            #[cfg(feature = "robust-aggregation")]
            RobustAggregation::MultiKrum {
                num_byzantine,
                num_selected,
            } => Krum::try_multi(*num_byzantine, *num_selected)?.aggregate(updates),
            #[cfg(feature = "robust-aggregation")]
            RobustAggregation::Bulyan {
                max_byzantine,
                profile,
            } => Bulyan::try_new(*max_byzantine, *profile)?.aggregate(updates),
            RobustAggregation::CenteredClipping {
                tau,
                max_iterations,
            } => {
                let cclip = CenteredClipping::try_new(*tau, *max_iterations, Vec::new())?;
                if let Some(ref center) = self.cclip_center {
                    cclip.aggregate_with_center(updates, center)
                } else {
                    cclip.aggregate(updates)
                }
            }
        }
        .map_err(RoundError::Aggregation)?;

        Ok((out, None))
    }

    fn pre_release_dp_debit_if_enabled(
        &mut self,
        session: &mut DataOpsSession,
        round_id: u64,
        ts_unix_nanos: u64,
        explicit_tau: Option<f32>,
        subsampling_q_override: Option<f64>,
    ) -> Result<Option<crate::artifacts::DpLedgerEntryV1>, RoundError> {
        let Some(acc) = self.dp_accountant.as_mut() else {
            return Ok(None);
        };
        if acc.guarantees_differential_privacy() {
            return Err(RoundError::Config(
                RoundConfigError::UnsupportedCertifiedDifferentialPrivacy,
            ));
        }
        let tau = explicit_tau.unwrap_or_else(|| match self.config.aggregation {
            RobustAggregation::CenteredClipping { tau, .. } => tau,
            _ => acc.profile().clipping_radius_tau,
        });
        let q = subsampling_q_override.unwrap_or_else(|| {
            let total = self.config.authorized_cohort.len().max(1) as f64;
            (self.admitted.len().max(1) as f64 / total).clamp(0.0, 1.0)
        });
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let (_, entry) = pre_release_debit_dp_accountant(
            &run_dir,
            self.run_id,
            round_id,
            &self.parent_digest.to_hex(),
            acc,
            tau,
            q,
            ts_unix_nanos,
        )
        .map_err(|err| match err {
            CryptoErrorOrIo::Crypto(c) => RoundError::PrivacyBudget(c),
            CryptoErrorOrIo::Io(e) => RoundError::Io(e),
        })?;
        Ok(Some(entry))
    }

    /// Whether this coordinator enforces a certified noise-injecting Differential Privacy
    /// mechanism (`false` in `v0.2` experimental hypothetical accounting mode, `ST-02`).
    pub const fn supports_certified_differential_privacy(&self) -> bool {
        false
    }

    /// Configure an **experimental hypothetical** Rényi Differential Privacy budget accountant
    /// backed by `dp/ledger.ndjson` (`ADR-0023` §4.2, `W12-03`, `ST-02`).
    ///
    /// **Important (`ST-02`):** Neither [`SwarmRoundCoordinator`] nor the ledger currently injects
    /// Poisson client subsampling or calibrated Gaussian noise. Fails closed with
    /// [`RoundConfigError::UnsupportedCertifiedDifferentialPrivacy`] if `profile.classification`
    /// requests [`swarm_torch_core::dp::PrivacyAccountingClassificationV1::VerifiedMechanismProtected`].
    ///
    /// Replays any already-committed entries from `run_dir/dp/ledger.ndjson` so a crash or restart
    /// never refunds previously debited hypothetical accounting budget.
    pub fn configure_dp_accountant(
        &mut self,
        run_dir: &std::path::Path,
        profile: swarm_torch_core::dp::DpMechanismProfileV1,
    ) -> Result<(), RoundError> {
        if profile.classification
            != swarm_torch_core::dp::PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly
            || !self.supports_certified_differential_privacy() && profile.guarantees_differential_privacy()
        {
            return Err(RoundError::Config(
                RoundConfigError::UnsupportedCertifiedDifferentialPrivacy,
            ));
        }
        let next_generation = self.coordinator_generation.checked_add(1).ok_or(
            RoundError::CoordinatorGenerationOverflow {
                generation: self.coordinator_generation,
            },
        )?;
        let acc =
            reconstruct_dp_accountant_from_ledger(run_dir, profile).map_err(|err| match err {
                CryptoErrorOrIo::Crypto(c) => RoundError::PrivacyBudget(c),
                CryptoErrorOrIo::Io(e) => RoundError::Io(e),
            })?;
        self.dp_accountant = Some(acc);
        self.coordinator_generation = next_generation;
        Ok(())
    }

    /// Borrow the active stateful Rényi Differential Privacy accountant, if configured.
    pub const fn dp_accountant(&self) -> Option<&swarm_torch_core::dp::RdpAccountantV1> {
        self.dp_accountant.as_ref()
    }

    /// Enable bounded-staleness asynchronous (`FedAsync`) delta mixing with drift compensation
    /// over the historical parameter anchor window (`ADR-0023` §4.1, `W12-03`).
    pub fn enable_fedasync(
        &mut self,
        policy: swarm_torch_core::aggregation::FedAsyncStalenessDescriptorV1,
        hinge_threshold_rounds: u64,
        clip_norm: Option<f32>,
    ) -> Result<(), RoundError> {
        policy.validate().map_err(RoundError::Aggregation)?;
        if let Some(tau) = clip_norm {
            if !tau.is_finite() || tau <= 0.0 {
                return Err(RoundError::Aggregation(CoreError::InvalidThreshold));
            }
        }
        let next_generation = self.coordinator_generation.checked_add(1).ok_or(
            RoundError::CoordinatorGenerationOverflow {
                generation: self.coordinator_generation,
            },
        )?;
        let latest = self
            .current_round
            .checked_sub(1)
            .ok_or(RoundError::RoundIdOverflow {
                round_id: self.current_round,
            })?;
        if !self.historical_anchors.contains_key(&latest) {
            self.historical_anchors
                .insert(latest, self.model.parameters().to_vec());
        }
        self.fedasync_policy = Some(policy);
        self.fedasync_hinge_threshold = hinge_threshold_rounds;
        self.fedasync_clip_norm = clip_norm;
        self.coordinator_generation = next_generation;
        Ok(())
    }

    /// Lookup retained server parameter vector for historical anchor round `round_id`, if still in window.
    pub fn historical_anchor_weights(&self, round_id: u64) -> Option<&[f32]> {
        self.historical_anchors.get(&round_id).map(|v| v.as_slice())
    }

    /// Legacy unverified `FedAsync` entrypoint — rejected fail-closed (`ST-01`).
    ///
    /// Use [`Self::prepare_verified_delivery`] -> [`ExpectedRoundContextV1::validate_fedasync_submission`]
    /// -> [`Self::commit_prepared_fedasync_submission`] -> [`Self::step_fedasync_verified_submission`]
    /// (or [`Self::step_fedasync_verified_delivery`]).
    #[deprecated(
        note = "unverified ingress fails closed; use step_fedasync_verified_delivery or the 3-stage pipeline"
    )]
    #[allow(deprecated)]
    pub fn step_fedasync_stale_submission(
        &mut self,
        _submission: PeerRoundSubmissionV1,
        _now_secs: u64,
        _subsampling_rate_q: f64,
        _session: &mut DataOpsSession,
        _ts_unix_nanos: u64,
    ) -> Result<
        (
            swarm_torch_core::aggregation::FedAsyncStaleMixOutcomeV1,
            RoundCommitSummaryV1,
        ),
        RoundError,
    > {
        Err(RoundError::UnverifiedSubmissionRejected)
    }

    /// Stage A -> B -> C convenience execution for a signed [`BundledGossipDeliveryV2`] targeting `FedAsync` (`ST-01`).
    pub fn step_fedasync_verified_delivery(
        &mut self,
        delivery_bytes: &[u8],
        max_delivery_bytes: usize,
        now_secs: u64,
        subsampling_rate_q: f64,
        session: &mut DataOpsSession,
        ts_unix_nanos: u64,
    ) -> Result<
        (
            swarm_torch_core::aggregation::FedAsyncStaleMixOutcomeV1,
            RoundCommitSummaryV1,
        ),
        RoundError,
    > {
        let prepared =
            self.prepare_verified_delivery(delivery_bytes, max_delivery_bytes, now_secs)?;
        let ctx = self.expected_round_context();
        let validated = ctx.validate_fedasync_submission(prepared)?;
        let verified = self.commit_prepared_fedasync_submission(validated)?;
        self.step_fedasync_verified_submission(verified, subsampling_rate_q, session, ts_unix_nanos)
    }

    /// Stage C (`FedAsync`): verify `coordinator_generation`, commit replay sequence atomically,
    /// and emit a sealed [`VerifiedFedAsyncSubmissionV1`] (`ST-01`).
    pub fn commit_prepared_fedasync_submission(
        &mut self,
        validated: ValidatedFedAsyncSubmissionV1,
    ) -> Result<VerifiedFedAsyncSubmissionV1, RoundError> {
        if validated.coordinator_generation != self.coordinator_generation {
            return Err(RoundError::StaleCoordinatorGeneration {
                expected: self.coordinator_generation,
                got: validated.coordinator_generation,
            });
        }
        self.replay
            .commit_prepared_admission(validated.replay_admission)
            .map_err(RoundError::ReplayRejected)?;
        Ok(VerifiedFedAsyncSubmissionV1 {
            coordinator_generation: validated.coordinator_generation,
            base_round_id: validated.base_round_id,
            sender: validated.sender,
            values: validated.values,
        })
    }

    /// Execute one bounded-staleness asynchronous (`FedAsync`) step using a sealed
    /// [`VerifiedFedAsyncSubmissionV1`] token (`ADR-0023` §4.1-4.2, `W12-03`, `ST-01`).
    pub fn step_fedasync_verified_submission(
        &mut self,
        verified: VerifiedFedAsyncSubmissionV1,
        subsampling_rate_q: f64,
        session: &mut DataOpsSession,
        ts_unix_nanos: u64,
    ) -> Result<
        (
            swarm_torch_core::aggregation::FedAsyncStaleMixOutcomeV1,
            RoundCommitSummaryV1,
        ),
        RoundError,
    > {
        if verified.coordinator_generation != self.coordinator_generation {
            return Err(RoundError::StaleCoordinatorGeneration {
                expected: self.coordinator_generation,
                got: verified.coordinator_generation,
            });
        }
        let policy = self
            .fedasync_policy
            .ok_or(RoundError::Aggregation(CoreError::InvalidThreshold))?;
        let latest_committed = self.current_round.saturating_sub(1);
        if verified.base_round_id > latest_committed {
            return Err(RoundError::FutureRoundRejected {
                submission_round: verified.base_round_id,
                current_round: latest_committed,
            });
        }
        let staleness = latest_committed - verified.base_round_id;
        let max_stale = policy.max_staleness_rounds.min(self.config.max_staleness);
        if staleness > max_stale {
            return Err(RoundError::StaleRoundRejected {
                submission_round: verified.base_round_id,
                current_round: latest_committed,
                max_staleness: max_stale,
            });
        }
        let w_base = self
            .historical_anchors
            .get(&verified.base_round_id)
            .ok_or(RoundError::StaleAnchorEvicted {
                base_round: verified.base_round_id,
                current_round: latest_committed,
            })?
            .clone();

        struct HingePolicy {
            inner: swarm_torch_core::aggregation::FedAsyncStalenessDescriptorV1,
            hinge: u64,
        }
        impl swarm_torch_core::aggregation::StalenessWeightingPolicy for HingePolicy {
            fn compute_mixing_alpha(
                &self,
                base_round_id: u64,
                current_round_id: u64,
            ) -> swarm_torch_core::Result<f32> {
                self.inner
                    .compute_hinge_mixing_alpha(base_round_id, current_round_id, self.hinge)
            }
        }

        let outcome = swarm_torch_core::aggregation::apply_fedasync_stale_delta_v1(
            &HingePolicy {
                inner: policy,
                hinge: self.fedasync_hinge_threshold,
            },
            verified.base_round_id,
            latest_committed,
            self.model.parameters(),
            &w_base,
            &verified.values,
            self.fedasync_clip_norm,
        )
        .map_err(RoundError::Aggregation)?;

        let mut candidate_model = self.model.clone();
        candidate_model
            .load_parameters(&outcome.updated_weights)
            .map_err(RoundError::Aggregation)?;

        let committed_round = self.current_round;
        let next_round = committed_round
            .checked_add(1)
            .ok_or(RoundError::RoundIdOverflow {
                round_id: committed_round,
            })?;
        let next_generation = self.coordinator_generation.checked_add(1).ok_or(
            RoundError::CoordinatorGenerationOverflow {
                generation: self.coordinator_generation,
            },
        )?;
        let next_historical_anchors = self.compute_next_historical_anchors(
            committed_round,
            candidate_model.parameters().to_vec(),
        );
        let mut frozen_raw_inputs = BTreeMap::new();
        frozen_raw_inputs.insert(verified.sender, verified.values);

        let checkpoint_asset_key = format!("checkpoint://round/{committed_round}");
        let node_key = format!("train/round_{committed_round}_fedasync");

        let plan = StagedRoundCommit {
            committed_round,
            next_round,
            next_generation,
            candidate_model,
            optimizer_snapshot: None,
            next_cclip_center: self.cclip_center.clone(),
            next_historical_anchors,
            next_historical_digests_without_new: self.historical_digests.clone(),
            next_bucketing_report: self.last_bucketing_report.clone(),
            checkpoint_asset_key,
            node_key,
            op_type: "fedasync_stale_step".to_string(),
            output_rows: Some(1),
            admitted_peers: vec![verified.sender],
            equivocated_peers: Vec::new(),
            frozen_raw_inputs,
            weight_mismatch_warnings: Vec::new(),
            fitness_score: None,
            staged_transform_audits: Vec::new(),
            dp_subsampling_q_override: Some(subsampling_rate_q),
            dp_explicit_tau: self.fedasync_clip_norm,
            emit_standard_telemetry: false,
            clear_round_admitted: false,
            ts_unix_nanos,
            telemetry_admitted_count: 1,
            telemetry_equivocated_count: 0,
        };

        let summary = self.commit_round_plan_v1(plan, session)?;
        Ok((outcome, summary))
    }

    /// Legacy unverified LoRA cohort aggregation entrypoint — rejected fail-closed (`ST-01`).
    ///
    /// Use [`Self::prepare_verified_delivery`] -> [`ExpectedRoundContextV1::validate_lora_submission`]
    /// -> [`Self::commit_prepared_lora_submission`] -> [`Self::aggregate_lora_cohort_verified_round`].
    #[deprecated(
        note = "unverified LoRA ingress fails closed; use aggregate_lora_cohort_verified_round with VerifiedLoraSubmissionV1"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn aggregate_lora_cohort_round(
        &mut self,
        _submissions: &[(PeerId, swarm_torch_models::lora::LoraAdapterPairV1)],
        _mode: swarm_torch_models::lora::LoraAggregationModeV1,
        _b_factor_aggregator: &dyn swarm_torch_core::aggregation::RobustAggregator,
        _base_weights_to_merge: Option<&mut [f32]>,
        _subsampling_rate_q: f64,
        _session: &mut DataOpsSession,
        _ts_unix_nanos: u64,
    ) -> Result<swarm_torch_models::lora::LoraAggregationOutcomeV1, RoundError> {
        Err(RoundError::UnverifiedSubmissionRejected)
    }

    /// Stage C (LoRA cohort): verify `coordinator_generation`, commit replay sequence atomically,
    /// and emit a sealed [`VerifiedLoraSubmissionV1`] (`ST-01`).
    pub fn commit_prepared_lora_submission(
        &mut self,
        validated: ValidatedLoraSubmissionV1,
    ) -> Result<VerifiedLoraSubmissionV1, RoundError> {
        if validated.coordinator_generation != self.coordinator_generation {
            return Err(RoundError::StaleCoordinatorGeneration {
                expected: self.coordinator_generation,
                got: validated.coordinator_generation,
            });
        }
        self.replay
            .commit_prepared_admission(validated.replay_admission)
            .map_err(RoundError::ReplayRejected)?;
        Ok(VerifiedLoraSubmissionV1 {
            coordinator_generation: validated.coordinator_generation,
            round_id: validated.round_id,
            sender: validated.sender,
            pair: validated.pair,
        })
    }

    /// Coordinate a cohort of cryptographically verified [`VerifiedLoraSubmissionV1`] tokens under
    /// generation binding, authorized cohort gating, quorum checks, staged base-weight merge, and
    /// pre-release DP ledger debit (`ADR-0024` §4, `W12-01`, `ST-01`).
    #[allow(clippy::too_many_arguments)]
    pub fn aggregate_lora_cohort_verified_round(
        &mut self,
        submissions: &[VerifiedLoraSubmissionV1],
        mode: swarm_torch_models::lora::LoraAggregationModeV1,
        b_factor_aggregator: &dyn swarm_torch_core::aggregation::RobustAggregator,
        base_weights_to_merge: Option<&mut [f32]>,
        subsampling_rate_q: f64,
        session: &mut DataOpsSession,
        ts_unix_nanos: u64,
    ) -> Result<swarm_torch_models::lora::LoraAggregationOutcomeV1, RoundError> {
        let mut seen_peers = BTreeSet::new();
        let mut pairs: Vec<(PeerId, LoraAdapterPairV1)> = Vec::with_capacity(submissions.len());
        for sub in submissions {
            if sub.coordinator_generation != self.coordinator_generation {
                return Err(RoundError::StaleCoordinatorGeneration {
                    expected: self.coordinator_generation,
                    got: sub.coordinator_generation,
                });
            }
            if sub.round_id != self.current_round {
                return Err(RoundError::StaleRoundRejected {
                    submission_round: sub.round_id,
                    current_round: self.current_round,
                    max_staleness: 0,
                });
            }
            if !self.config.authorized_cohort.contains(&sub.sender) {
                return Err(RoundError::ReplayRejected(ReplayError::UnauthorizedPeer {
                    peer: sub.sender,
                }));
            }
            if !seen_peers.insert(sub.sender) {
                return Err(RoundError::Aggregation(CoreError::InvalidGradient));
            }
            pairs.push((sub.sender, sub.pair.clone()));
        }
        if pairs.len() < self.config.min_quorum {
            return Err(RoundError::InsufficientQuorum {
                admitted: pairs.len(),
                min_quorum: self.config.min_quorum,
            });
        }

        let outcome =
            swarm_torch_models::lora::aggregate_lora_adapters_v1(&pairs, mode, b_factor_aggregator)
                .map_err(|_| RoundError::Aggregation(CoreError::InvalidGradient))?;

        // Prepare & validate the candidate base-weight merge in a staging copy BEFORE attempting
        // durable DP pre-debit, and commit the mutation to `base_weights_to_merge` ONLY AFTER the
        // debit succeeds (`ST-02` failure atomicity).
        let mut staged_base_weights = match base_weights_to_merge.as_deref() {
            Some(base_w) => {
                if base_w.iter().any(|v| v.is_nan()) {
                    return Err(RoundError::Aggregation(CoreError::NanDetected));
                }
                if base_w.iter().any(|v| !v.is_finite()) {
                    return Err(RoundError::Aggregation(CoreError::InfDetected));
                }
                let mut staged = base_w.to_vec();
                match &outcome {
                    swarm_torch_models::lora::LoraAggregationOutcomeV1::FloraStacked { stacked } => {
                        stacked
                            .apply_exact_to_base_weights(&mut staged)
                            .map_err(|_| RoundError::Aggregation(CoreError::InvalidGradient))?;
                    }
                    swarm_torch_models::lora::LoraAggregationOutcomeV1::SingleFactorAggregated {
                        adapter,
                    }
                    | swarm_torch_models::lora::LoraAggregationOutcomeV1::BiasedFactorFedAvg {
                        adapter,
                        ..
                    } => {
                        adapter
                            .apply_to_base_weights(&mut staged)
                            .map_err(|_| RoundError::Aggregation(CoreError::InvalidGradient))?;
                    }
                }
                Some(staged)
            }
            None => None,
        };

        let next_generation = self.coordinator_generation.checked_add(1).ok_or(
            RoundError::CoordinatorGenerationOverflow {
                generation: self.coordinator_generation,
            },
        )?;

        // Fail-closed DP debit before committing base-weight merge or releasing aggregated factors.
        self.pre_release_dp_debit_if_enabled(
            session,
            self.current_round,
            ts_unix_nanos,
            None,
            Some(subsampling_rate_q),
        )?;

        if let (Some(base_w), Some(staged)) = (base_weights_to_merge, staged_base_weights.take()) {
            base_w.copy_from_slice(&staged);
        }
        self.coordinator_generation = next_generation;

        Ok(outcome)
    }
}

fn decode_prepared_payload_v2(
    inner: &InnerUpdateV2,
    originator: PeerId,
) -> Result<PreparedPayloadV2, RoundError> {
    let expected_run = RunId::from_bytes(inner.run_id);
    let expected_base = CheckpointContentDigestV1(inner.base_checkpoint_digest);
    match inner.compression_schema_version {
        COMPRESSION_SCHEMA_DENSE_F32_LE_V1 => {
            let expected_bytes = (inner.num_elements as usize)
                .checked_mul(4)
                .ok_or_else(|| {
                    RoundError::InvalidWirePayload("num_elements byte length overflow".to_string())
                })?;
            if inner.encoded_payload.len() != expected_bytes {
                return Err(RoundError::InvalidWirePayload(format!(
                    "dense payload length mismatch: expected {expected_bytes}, got {}",
                    inner.encoded_payload.len()
                )));
            }
            let mut values = Vec::with_capacity(inner.num_elements as usize);
            for (idx, chunk) in inner.encoded_payload.chunks_exact(4).enumerate() {
                let mut arr = [0u8; 4];
                arr.copy_from_slice(chunk);
                let val = f32::from_le_bytes(arr);
                if !val.is_finite() {
                    return Err(RoundError::NonFiniteInput {
                        sender: originator,
                        coord_index: idx,
                    });
                }
                values.push(val);
            }
            let canonical_payload_digest = canonical_dense_submission_digest_v2(
                &inner.swarm_instance_id,
                expected_run,
                inner.round_id,
                &expected_base,
                &inner.context_policy_digest,
                originator,
                inner.payload_kind,
                &values,
            );
            Ok(PreparedPayloadV2::Dense {
                values,
                canonical_payload_digest,
            })
        }
        COMPRESSION_SCHEMA_COMPRESSED_GRADIENT_V1 => {
            let compressed: CompressedGradient = serde_json::from_slice(&inner.encoded_payload)
                .map_err(|e| {
                    RoundError::InvalidWirePayload(format!("invalid compressed gradient JSON: {e}"))
                })?;
            compressed
                .validate_shape_and_bounds(inner.num_elements as usize)
                .map_err(|e| {
                    RoundError::InvalidWirePayload(format!(
                        "compressed gradient bounds check failed: {e:?}"
                    ))
                })?;
            if compressed.num_elements != inner.num_elements as usize {
                return Err(RoundError::DimensionMismatch {
                    expected: inner.num_elements as usize,
                    got: compressed.num_elements,
                });
            }
            let values = compressed
                .decompress_with_limit(inner.num_elements as usize)
                .map_err(|e| {
                    RoundError::InvalidWirePayload(format!(
                        "compressed gradient decompression failed: {e:?}"
                    ))
                })?;
            if values.len() != inner.num_elements as usize {
                return Err(RoundError::DimensionMismatch {
                    expected: inner.num_elements as usize,
                    got: values.len(),
                });
            }
            for (idx, &val) in values.iter().enumerate() {
                if !val.is_finite() {
                    return Err(RoundError::NonFiniteInput {
                        sender: originator,
                        coord_index: idx,
                    });
                }
            }
            let canonical_payload_digest = canonical_dense_submission_digest_v2(
                &inner.swarm_instance_id,
                expected_run,
                inner.round_id,
                &expected_base,
                &inner.context_policy_digest,
                originator,
                inner.payload_kind,
                &values,
            );
            Ok(PreparedPayloadV2::Dense {
                values,
                canonical_payload_digest,
            })
        }
        COMPRESSION_SCHEMA_LORA_PAIR_V1 => {
            let decoded: LoraAdapterPairV1 = serde_json::from_slice(&inner.encoded_payload)
                .map_err(|e| {
                    RoundError::InvalidWirePayload(format!("invalid LoRA adapter JSON: {e}"))
                })?;
            let pair = LoraAdapterPairV1::try_new(
                decoded.descriptor,
                decoded.d_in,
                decoded.d_out,
                decoded.a_weights,
                decoded.b_weights,
            )
            .map_err(|e| {
                RoundError::InvalidWirePayload(format!("invalid LoRA adapter pair: {e:?}"))
            })?;
            let total_coords = pair.a_weights.len() + pair.b_weights.len();
            if total_coords != inner.num_elements as usize {
                return Err(RoundError::DimensionMismatch {
                    expected: inner.num_elements as usize,
                    got: total_coords,
                });
            }
            Ok(PreparedPayloadV2::Lora { pair })
        }
        other => Err(RoundError::InvalidWirePayload(format!(
            "unsupported compression_schema_version {other}"
        ))),
    }
}

fn aggregate_registry_certified_fedavg(
    admitted_peers: &[PeerId],
    updates: &[GradientUpdate],
    peer_weights: &BTreeMap<PeerId, u64>,
    max_weight_share: f64,
) -> Result<Vec<f32>, RoundError> {
    if updates.is_empty() || admitted_peers.len() != updates.len() {
        return Err(RoundError::Aggregation(CoreError::InsufficientUpdates));
    }
    let dim = updates[0].gradients.len();
    let mut raw_weights = Vec::with_capacity(admitted_peers.len());
    let mut raw_sum = 0.0f64;

    for peer in admitted_peers {
        let &w = peer_weights.get(peer).ok_or(RoundError::Config(
            RoundConfigError::MissingOrZeroCertifiedWeight(*peer),
        ))?;
        if w == 0 {
            return Err(RoundError::Config(
                RoundConfigError::MissingOrZeroCertifiedWeight(*peer),
            ));
        }
        let wf = w as f64;
        raw_weights.push(wf);
        raw_sum += wf;
    }

    if !raw_sum.is_finite() || raw_sum <= 0.0 {
        return Err(RoundError::Aggregation(CoreError::InvalidGradient));
    }

    let mut capped_weights = Vec::with_capacity(raw_weights.len());
    let mut capped_sum = 0.0f64;
    for wf in raw_weights {
        let share = (wf / raw_sum).min(max_weight_share);
        capped_weights.push(share);
        capped_sum += share;
    }

    if !capped_sum.is_finite() || capped_sum <= 0.0 {
        return Err(RoundError::Aggregation(CoreError::InvalidGradient));
    }

    let mut normalized_weights = Vec::with_capacity(capped_weights.len());
    for share in capped_weights {
        normalized_weights.push(share / capped_sum);
    }

    let mut acc = vec![0.0f64; dim];
    for (update, &alpha) in updates.iter().zip(normalized_weights.iter()) {
        if update.gradients.len() != dim {
            return Err(RoundError::Aggregation(CoreError::InvalidGradient));
        }
        for (slot, &val) in acc.iter_mut().zip(update.gradients.iter()) {
            if val.is_nan() {
                return Err(RoundError::Aggregation(CoreError::NanDetected));
            }
            if val.is_infinite() {
                return Err(RoundError::Aggregation(CoreError::InfDetected));
            }
            *slot += alpha * (val as f64);
        }
    }

    let mut out = Vec::with_capacity(dim);
    for val_f64 in acc {
        let val_f32 = val_f64 as f32;
        if !val_f32.is_finite() {
            return Err(RoundError::Aggregation(CoreError::InfDetected));
        }
        out.push(val_f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use swarm_torch_net::gossip::DEFAULT_MAX_GOSSIP_DELIVERY_BYTES;

    use crate::artifacts::{ArtifactWriteProfile, LockAcquisitionPolicy};
    use swarm_torch_models::optim::SgdOptimizer;
    use swarm_torch_models::simple::LinearModel;

    fn linear_model_2_1(params: &[f32]) -> LinearModel {
        let mut m = LinearModel::new(2, 1).unwrap();
        m.load_parameters(params).unwrap();
        m
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("swarmtorch_coordinator_{prefix}_{pid}_{nanos}"));
        dir
    }

    struct TestPeer {
        auth: swarm_torch_core::crypto::MessageAuth,
        peer_id: PeerId,
    }

    impl TestPeer {
        fn new(seed: u8) -> Self {
            let kp = swarm_torch_core::crypto::KeyPair::from_seed([seed; 32]).unwrap();
            let peer_id = kp.peer_id();
            Self {
                auth: swarm_torch_core::crypto::MessageAuth::new(kp),
                peer_id,
            }
        }

        fn establish_session<M: SwarmModel + Clone>(
            &self,
            coord: &mut SwarmRoundCoordinator<M>,
            epoch: u64,
            nonce_seed: u8,
            now_secs: u32,
        ) -> [u8; 32] {
            let challenge = SessionHandshakeChallengeV2 {
                swarm_instance_id: coord.swarm_instance_id(),
                run_id: *coord.run_id().as_bytes(),
                policy_digest: coord.context_policy_digest(),
                receiver_peer: coord.coordinator_peer_id(),
                sender_peer: self.peer_id,
                requested_epoch: epoch,
                initial_sequence: 0,
                receiver_nonce: [nonce_seed; 32],
                issued_at_secs: now_secs,
            };
            let session_id = challenge.derive_session_id().unwrap();
            coord.register_session_challenge_v2(challenge).unwrap();
            let sig = self.auth.sign_session_handshake_v2(&challenge).unwrap();
            coord
                .complete_session_handshake_v2(self.auth.key_pair().public_key(), &sig, now_secs)
                .unwrap();
            session_id
        }

        #[allow(clippy::too_many_arguments)]
        fn build_dense_delivery<M: SwarmModel + Clone>(
            &self,
            coord: &SwarmRoundCoordinator<M>,
            session_id: [u8; 32],
            epoch: u64,
            seq: u64,
            ts_secs: u32,
            round_id: u64,
            base_digest: CheckpointContentDigestV1,
            payload_kind: PayloadKindV1,
            values: &[f32],
            reported_examples: u64,
        ) -> Vec<u8> {
            let mut raw = Vec::with_capacity(values.len() * 4);
            for &v in values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let inner = InnerUpdateV2 {
                originator_public_key: *self.auth.key_pair().public_key(),
                swarm_instance_id: coord.swarm_instance_id(),
                run_id: *coord.run_id().as_bytes(),
                round_id,
                contribution_id: seq,
                base_checkpoint_digest: *base_digest.as_bytes(),
                context_policy_digest: coord.context_policy_digest(),
                payload_kind,
                compression_schema_version: COMPRESSION_SCHEMA_DENSE_F32_LE_V1,
                num_elements: values.len() as u32,
                reported_examples,
                encoded_payload: raw,
                originator_signature: Vec::new(),
            };
            BundledGossipDeliveryV2::new_signed(
                &self.auth,
                inner,
                coord.coordinator_peer_id(),
                session_id,
                epoch,
                seq,
                ts_secs,
            )
            .to_bytes()
            .unwrap()
        }

        #[allow(clippy::too_many_arguments)]
        fn build_lora_delivery<M: SwarmModel + Clone>(
            &self,
            coord: &SwarmRoundCoordinator<M>,
            session_id: [u8; 32],
            epoch: u64,
            seq: u64,
            ts_secs: u32,
            round_id: u64,
            base_digest: CheckpointContentDigestV1,
            pair: &LoraAdapterPairV1,
        ) -> Vec<u8> {
            let encoded_payload = encode_lora_update_payload_v2(pair).unwrap();
            let inner = InnerUpdateV2 {
                originator_public_key: *self.auth.key_pair().public_key(),
                swarm_instance_id: coord.swarm_instance_id(),
                run_id: *coord.run_id().as_bytes(),
                round_id,
                contribution_id: seq,
                base_checkpoint_digest: *base_digest.as_bytes(),
                context_policy_digest: coord.context_policy_digest(),
                payload_kind: PayloadKindV1::ModelDeltaV1,
                compression_schema_version: COMPRESSION_SCHEMA_LORA_PAIR_V1,
                num_elements: (pair.a_weights.len() + pair.b_weights.len()) as u32,
                reported_examples: 0,
                encoded_payload,
                originator_signature: Vec::new(),
            };
            BundledGossipDeliveryV2::new_signed(
                &self.auth,
                inner,
                coord.coordinator_peer_id(),
                session_id,
                epoch,
                seq,
                ts_secs,
            )
            .to_bytes()
            .unwrap()
        }
    }

    fn verify_lora_delivery<M: SwarmModel + Clone>(
        coord: &mut SwarmRoundCoordinator<M>,
        peer: &TestPeer,
        session_id: [u8; 32],
        epoch: u64,
        seq: u64,
        ts_secs: u32,
        pair: &LoraAdapterPairV1,
    ) -> Result<VerifiedLoraSubmissionV1, RoundError> {
        let round_id = coord.current_round();
        let base_digest = coord.parent_digest();
        let bytes = peer.build_lora_delivery(
            coord,
            session_id,
            epoch,
            seq,
            ts_secs,
            round_id,
            base_digest,
            pair,
        );
        let prepared = coord.prepare_verified_delivery(
            &bytes,
            DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
            ts_secs as u64,
        )?;
        let ctx = coord.expected_round_context();
        let validated = ctx.validate_lora_submission(prepared)?;
        coord.commit_prepared_lora_submission(validated)
    }

    fn peer(seed: u8) -> PeerId {
        let kp = swarm_torch_core::crypto::KeyPair::from_seed([seed; 32]).unwrap();
        PeerId::from_public_key(kp.public_key())
    }

    #[test]
    fn coordinator_rejects_future_round_without_underflow_and_enforces_staleness() {
        let base = temp_dir("future_and_stale");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x11; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(1);
        let p2 = TestPeer::new(2);
        let cohort: BTreeSet<PeerId> = [p1.peer_id, p2.peer_id].into_iter().collect();
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::default(),
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: cohort,
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let model = linear_model_2_1(&[1.0, 2.0, 0.5]);
        let mut coord =
            SwarmRoundCoordinator::initialize(run_id, config, model, &mut session, 1_000).unwrap();
        let s1 = p1.establish_session(&mut coord, 1, 0xA1, 100);

        // Submit future round u64::MAX -> must return FutureRoundRejected without underflow panic
        // and WITHOUT consuming replay sequence 1!
        let future_bytes = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            u64::MAX,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        let err = coord
            .submit_verified_delivery(&future_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap_err();
        assert!(matches!(
            err,
            RoundError::FutureRoundRejected {
                submission_round: u64::MAX,
                current_round: 1
            }
        ));
        assert_eq!(coord.admitted_count(), 0);

        // Submit stale round 0 when max_staleness == 0 -> StaleRoundRejected!
        let stale_bytes = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            0,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        let err = coord
            .submit_verified_delivery(&stale_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap_err();
        assert!(matches!(err, RoundError::StaleRoundRejected { .. }));

        // Sequence 1 was NOT consumed by Stage A/B failures, so valid round 1 with seq=1 succeeds!
        let valid_bytes = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        assert!(matches!(
            coord
                .submit_verified_delivery(&valid_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap(),
            SubmissionDecision::Admitted {
                admitted_count: 1,
                ..
            }
        ));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn coordinator_handles_idempotent_duplicate_and_revokes_equivocating_peer() {
        let base = temp_dir("duplicate_and_equivocation");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x22; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(1);
        let p2 = TestPeer::new(2);
        let cohort: BTreeSet<PeerId> = [p1.peer_id, p2.peer_id].into_iter().collect();
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::default(),
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: cohort,
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let model = linear_model_2_1(&[1.0, 2.0, 0.5]);
        let mut coord =
            SwarmRoundCoordinator::initialize(run_id, config, model, &mut session, 1_000).unwrap();
        let s1 = p1.establish_session(&mut coord, 1, 0xB1, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0xB2, 100);

        let sub1_seq1 = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.2, 0.4, 0.6],
            0,
        );
        assert_eq!(
            coord
                .submit_verified_delivery(&sub1_seq1, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap(),
            SubmissionDecision::Admitted {
                admitted_count: 1,
                min_quorum: 2
            }
        );

        // Exact wire packet replay (seq = 1) must be rejected by ReplayProtection!
        let replay_err = coord
            .submit_verified_delivery(&sub1_seq1, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap_err();
        assert!(matches!(
            replay_err,
            RoundError::ReplayRejected(ReplayError::Replay { seq: 1, .. })
        ));

        // Identical round content re-sent under fresh seq=2 -> DuplicateIdempotentAck (still 1 vote!).
        let sub1_seq2 = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            2,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.2, 0.4, 0.6],
            0,
        );
        assert_eq!(
            coord
                .submit_verified_delivery(&sub1_seq2, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap(),
            SubmissionDecision::DuplicateIdempotentAck
        );
        assert_eq!(coord.admitted_count(), 1);

        // Admit p2 so quorum (2) is reached.
        let sub2 = p2.build_dense_delivery(
            &coord,
            s2,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.4, 0.2, 0.0],
            0,
        );
        coord
            .submit_verified_delivery(&sub2, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap();
        assert!(coord.has_quorum());

        // Now p1 equivocates by sending a different vector in round 1!
        let conflicting_sub1 = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            3,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[-9.0, -9.0, -9.0],
            0,
        );
        let err = coord
            .submit_verified_delivery(&conflicting_sub1, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap_err();
        assert!(matches!(err, RoundError::EquivocationRejected { .. }));
        assert_eq!(
            coord.admitted_count(),
            1,
            "equivocating peer's earlier vote must be revoked"
        );
        assert!(!coord.has_quorum());

        // Attempting to finalize_round after equivocation broke quorum must fail closed!
        let fin_err = coord
            .finalize_round::<SgdOptimizer<LinearModel>, DummyEvaluator>(
                &[],
                None,
                None,
                &mut session,
                2_000,
            )
            .unwrap_err();
        assert!(matches!(
            fin_err,
            RoundError::InsufficientQuorum {
                admitted: 1,
                min_quorum: 2
            }
        ));

        let _ = fs::remove_dir_all(&base);
    }

    struct DummyEvaluator;
    impl FitnessFunction for DummyEvaluator {
        type Model = LinearModel;
        type Error = core::convert::Infallible;
        fn evaluate(&self, model: &Self::Model) -> Result<f32, Self::Error> {
            Ok(model.parameters().iter().sum())
        }
    }

    struct MutatingClipTransform;
    impl UpdateTransform for MutatingClipTransform {
        fn transform(&self, mut update: GradientUpdate) -> GradientUpdate {
            for g in &mut update.gradients {
                *g = g.clamp(-0.5, 0.5);
            }
            update
        }
        fn name(&self) -> &str {
            "clip_0_5"
        }
        fn is_core_trusted(&self) -> bool {
            true
        }
    }

    #[test]
    fn coordinator_commits_cas_checkpoint_preserves_frozen_inputs_and_resumes() {
        let base = temp_dir("commit_and_resume");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x33; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(10);
        let p2 = TestPeer::new(20);
        let cohort: BTreeSet<PeerId> = [p1.peer_id, p2.peer_id].into_iter().collect();

        // Reject RegistryCertified paired with TrimmedMean
        let mut bad_weights = BTreeMap::new();
        bad_weights.insert(p1.peer_id, 100);
        bad_weights.insert(p2.peer_id, 300);
        let bad_cfg = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::default(),
            aggregation: RobustAggregation::TrimmedMean { trim_ratio: 0.2 },
            weighting_policy: WeightingPolicyV1::RegistryCertified {
                peer_weights: bad_weights.clone(),
                max_weight_share: 0.8,
            },
            authorized_cohort: cohort.clone(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        assert_eq!(
            bad_cfg.validate().unwrap_err(),
            RoundConfigError::IncompatibleWeightingPolicy
        );

        // Valid RegistryCertified with FedAvg
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::RegistryCertified {
                peer_weights: bad_weights,
                max_weight_share: 0.75,
            },
            authorized_cohort: cohort,
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let model = linear_model_2_1(&[1.0, 2.0, 0.5]);
        let mut coord =
            SwarmRoundCoordinator::initialize(run_id, config.clone(), model, &mut session, 1_000)
                .unwrap();
        let s1 = p1.establish_session(&mut coord, 1, 0xC1, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0xC2, 100);

        // p1 sends raw delta [2.0, 2.0, 2.0] with spoofed reported_examples = 999_999
        let sub1 = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[2.0, 2.0, 2.0],
            999_999,
        );
        coord
            .submit_verified_delivery(&sub1, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap();

        // p2 sends raw delta [0.2, 0.2, 0.2]
        let sub2 = p2.build_dense_delivery(
            &coord,
            s2,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.2, 0.2, 0.2],
            300,
        );
        coord
            .submit_verified_delivery(&sub2, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap();

        let clip = MutatingClipTransform;
        let transforms: [&dyn UpdateTransform; 1] = [&clip];
        let summary = coord
            .finalize_round::<SgdOptimizer<LinearModel>, DummyEvaluator>(
                &transforms,
                None,
                Some(&DummyEvaluator),
                &mut session,
                2_000,
            )
            .unwrap();

        // Frozen raw input for p1 must still be [2.0, 2.0, 2.0] even though MutatingClipTransform clamped to 0.5!
        assert_eq!(summary.frozen_raw_inputs[&p1.peer_id], vec![2.0, 2.0, 2.0]);
        assert_eq!(summary.weight_mismatch_warnings.len(), 1);

        // Certified weights: p1=100 (0.25) * 0.5 + p2=300 (0.75) * 0.2 = 0.125 + 0.15 = 0.275
        let expected_w0 = 1.0f32 + 0.275f32;
        assert!((coord.model().parameters()[0] - expected_w0).abs() < 1e-6);

        // Commit and close session, then test resume from CAS!
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let cloned_sink = std::sync::Arc::clone(session.sink());
        let receipt = session.commit_and_close(None).unwrap();
        assert!(receipt.sealed_generation >= 2);

        // Post-seal write on cloned_sink must fail closed with PermissionDenied!
        let late_err = cloned_sink
            .emit_event(&EventRecord {
                schema_version: 1,
                ts_unix_nanos: 3_000,
                trace_id: TraceId::from_bytes([1; 16]),
                span_id: None,
                name: "late_event".to_string(),
                attrs: AttrMap::new(),
            })
            .unwrap_err();
        assert_eq!(late_err.kind(), io::ErrorKind::PermissionDenied);

        // Resume session and coordinator from disk
        let (resumed_session, resume_state) = DataOpsSession::resume(
            &run_dir,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let resume_state = resume_state.expect("resume state must exist");
        assert_eq!(resume_state.round_id, 1);

        #[allow(deprecated)]
        let mut resumed_coord: SwarmRoundCoordinator<LinearModel> =
            SwarmRoundCoordinator::from_resume_state(
                config,
                LinearModel::new(2, 1).unwrap(),
                &resume_state,
                &resumed_session,
            )
            .unwrap();
        assert_eq!(resumed_coord.current_round(), 2);
        assert_eq!(
            resumed_coord.model().parameters(),
            coord.model().parameters()
        );

        // Post-resume deliveries using pre-crash session s1 MUST fail closed with RecoveryRequired!
        let stale_session_bytes = p1.build_dense_delivery(
            &resumed_coord,
            s1,
            1,
            2,
            110,
            2,
            resumed_coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        let rec_err = resumed_coord
            .submit_verified_delivery(&stale_session_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 110)
            .unwrap_err();
        assert!(matches!(
            rec_err,
            RoundError::ReplayRejected(ReplayError::RecoveryRequired { .. })
        ));

        // Establishing a fresh session handshake at epoch=2 after restart admits Round 2!
        let s1_r2 = p1.establish_session(&mut resumed_coord, 2, 0xD1, 110);
        let r2_bytes = p1.build_dense_delivery(
            &resumed_coord,
            s1_r2,
            2,
            1,
            110,
            2,
            resumed_coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        assert!(resumed_coord
            .submit_verified_delivery(&r2_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 110)
            .is_ok());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn coordinator_stateful_centered_clipping_and_s_bucketing_across_rounds() {
        use swarm_torch_core::aggregation::BucketRemainderPolicy;

        let base = temp_dir("cclip_bucketing");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x66; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let peers: Vec<TestPeer> = (1..=6u8).map(TestPeer::new).collect();
        let cohort: BTreeSet<PeerId> = peers.iter().map(|p| p.peer_id).collect();
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::CenteredClipping {
                tau: 0.5,
                max_iterations: 3,
            },
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: cohort,
            min_quorum: 6,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let model = linear_model_2_1(&[1.0, 2.0, 0.5]);
        let mut coord =
            SwarmRoundCoordinator::initialize(run_id, config, model, &mut session, 1_000).unwrap();
        coord.set_bucketing(Some(BucketingConfig {
            bucket_size: 2,
            byzantine_count: 1,
            round_seed: [0x99; 32],
            remainder_policy: BucketRemainderPolicy::StrictExactMultiple,
        }));
        assert!(coord.cclip_center().is_none());

        let sessions: Vec<[u8; 32]> = peers
            .iter()
            .enumerate()
            .map(|(i, p)| p.establish_session(&mut coord, 1, 0x30 + i as u8, 100))
            .collect();

        // Round 1: 5 honest peers submit [0.2, 0.2, 0.2], 1 Byzantine peer submits [100.0, 100.0, 100.0].
        for (i, p) in peers.iter().enumerate() {
            let vals = if i < 5 {
                vec![0.2, 0.2, 0.2]
            } else {
                vec![100.0, 100.0, 100.0]
            };
            let bytes = p.build_dense_delivery(
                &coord,
                sessions[i],
                1,
                1,
                100,
                1,
                coord.parent_digest(),
                PayloadKindV1::ModelDeltaV1,
                &vals,
                0,
            );
            coord
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap();
        }
        let _r1 = coord
            .finalize_round::<SgdOptimizer<LinearModel>, DummyEvaluator>(
                &[],
                None,
                None,
                &mut session,
                2_000,
            )
            .unwrap();
        let center_r1 = coord
            .cclip_center()
            .expect("cclip_center must be recorded after round 1")
            .to_vec();
        assert_eq!(center_r1.len(), 3);
        let b_rep = coord
            .last_bucketing_report()
            .expect("bucketing report must be recorded");
        assert_eq!(b_rep.num_buckets, 3);
        assert_eq!(b_rep.effective_byzantine_bound, 1);

        // Round 2 clips around center_r1!
        for (i, p) in peers.iter().enumerate() {
            let vals = if i < 5 {
                vec![0.22, 0.22, 0.22]
            } else {
                vec![-500.0, -500.0, -500.0]
            };
            let bytes = p.build_dense_delivery(
                &coord,
                sessions[i],
                1,
                2,
                110,
                2,
                coord.parent_digest(),
                PayloadKindV1::ModelDeltaV1,
                &vals,
                0,
            );
            coord
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 110)
                .unwrap();
        }
        let _r2 = coord
            .finalize_round::<SgdOptimizer<LinearModel>, DummyEvaluator>(
                &[],
                None,
                None,
                &mut session,
                3_000,
            )
            .unwrap();
        let center_r2 = coord.cclip_center().unwrap();
        assert!((center_r2[0] - 0.22).abs() < 0.35);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn fedasync_anchor_window_drift_compensation_byzantine_clipping_and_eviction() {
        let base = temp_dir("fedasync_drift");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x77; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(1);
        let p2 = TestPeer::new(2);
        let cohort: BTreeSet<PeerId> = [p1.peer_id, p2.peer_id].into_iter().collect();
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: cohort,
            min_quorum: 2,
            max_staleness: 1,
            max_clock_skew_secs: 120,
        };

        // Round 0 genesis parameters = [1.0, 2.0, 0.5]
        let model = linear_model_2_1(&[1.0, 2.0, 0.5]);
        let mut coord =
            SwarmRoundCoordinator::initialize(run_id, config, model, &mut session, 1_000).unwrap();
        let digest_r0 = coord.parent_digest();
        coord
            .enable_fedasync(
                swarm_torch_core::aggregation::FedAsyncStalenessDescriptorV1 {
                    base_alpha: 0.5,
                    max_staleness_rounds: 1,
                    polynomial_decay_exponent: 1.0,
                },
                0,
                Some(1.0),
            )
            .unwrap();
        let s1 = p1.establish_session(&mut coord, 1, 0xE1, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0xE2, 100);

        // Advance synchronous round 1: move weights to [2.0, 3.0, 1.5]
        for (p, s) in [(&p1, s1), (&p2, s2)] {
            let bytes = p.build_dense_delivery(
                &coord,
                s,
                1,
                1,
                100,
                1,
                digest_r0,
                PayloadKindV1::ModelDeltaV1,
                &[1.0, 1.0, 1.0],
                0,
            );
            coord
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap();
        }
        let _r1 = coord
            .finalize_round::<SgdOptimizer<LinearModel>, DummyEvaluator>(
                &[],
                None,
                None,
                &mut session,
                2_000,
            )
            .unwrap();

        // Now latest committed round is 1 (w_current = [2.0, 3.0, 1.5]). Round 0 anchor is still in window (max_staleness = 1).
        // Byzantine peer 2 submits stale update anchored at round 0 with giant norm [1000.0, 0.0, 0.0].
        // Pre-drift norm clipping MUST bound stale_delta to norm <= 1.0 BEFORE adding server drift (w_0 - w_1)!
        let stale_p2 = p2.build_dense_delivery(
            &coord,
            s2,
            1,
            2,
            105,
            0,
            digest_r0,
            PayloadKindV1::ModelDeltaV1,
            &[1000.0, 0.0, 0.0],
            0,
        );
        let (outcome, _sum2) = coord
            .step_fedasync_verified_delivery(
                &stale_p2,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                105,
                0.5,
                &mut session,
                3_000,
            )
            .unwrap();
        assert!(outcome.was_clipped);
        assert!((outcome.clipped_delta_l2_norm - 1.0).abs() < 1e-4);
        assert_eq!(outcome.staleness_rounds, 1);

        // After advancing one more step, round 0 anchor was evicted (latest_committed = 2, cutoff = 1).
        // Attempting to reuse round 0 fails in Stage B WITHOUT consuming p1's replay sequence 2!
        assert!(coord.historical_anchor_weights(0).is_none());
        let stale_p1 = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            2,
            106,
            0,
            digest_r0,
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        let stale_err = coord
            .step_fedasync_verified_delivery(
                &stale_p1,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                106,
                0.5,
                &mut session,
                4_000,
            )
            .unwrap_err();
        assert!(matches!(stale_err, RoundError::StaleRoundRejected { .. }));

        // Prove p1's sequence 2 was NOT consumed by the rejected stale FedAsync delivery:
        let current_p1 = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            2,
            106,
            3,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        assert!(coord
            .submit_verified_delivery(&current_p1, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 106)
            .is_ok());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn dp_pre_release_ledger_debit_is_non_refundable_and_idempotent_across_crash_recovery() {
        use swarm_torch_core::crypto::CryptoError;
        use swarm_torch_core::dp::{
            DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1,
            PrivacyAccountingClassificationV1,
        };

        let base = temp_dir("dp_ledger_recovery");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x88; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let p1 = peer(1);
        let p2 = peer(2);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::CenteredClipping {
                tau: 1.0,
                max_iterations: 2,
            },
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1, p2].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };

        let dp_profile = DpMechanismProfileV1 {
            unit: DpUnitV1::ClientLevel,
            adjacency: DpAdjacencyV1::AddRemoveOne,
            clipping_radius_tau: 1.0,
            noise_multiplier: 1.5,
            target_delta: 1e-5,
            max_epsilon: 5.0,
            conversion_formula: DpConversionFormulaV1::Balle2020Tight,
            classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
        };

        let model = linear_model_2_1(&[1.0, 2.0, 0.5]);
        let mut coord =
            SwarmRoundCoordinator::initialize(run_id, config.clone(), model, &mut session, 1_000)
                .unwrap();
        coord.configure_dp_accountant(&run_dir, dp_profile).unwrap();

        // Simulate crash AFTER durable pre-release debit for round 1, BEFORE checkpoint publish:
        let receipt_hex =
            crate::artifacts::derive_dp_receipt_id(run_id, 1, &coord.parent_digest().to_hex(), 0);
        let mut crashed_acc = *coord.dp_accountant().unwrap();
        let (newly_appended, crashed_entry) = crate::artifacts::pre_release_debit_dp_accountant(
            &run_dir,
            run_id,
            1,
            &coord.parent_digest().to_hex(),
            &mut crashed_acc,
            1.0,
            1.0,
            1_500,
        )
        .unwrap();
        assert!(newly_appended);
        assert_eq!(crashed_entry.receipt_id, receipt_hex);

        // Idempotent retry of identical receipt_id does NOT double-debit!
        let replayed =
            crate::artifacts::append_dp_release_record_durable(&run_dir, &crashed_entry).unwrap();
        assert!(!replayed);
        assert_eq!(
            crate::artifacts::read_dp_ledger_entries(&run_dir)
                .unwrap()
                .len(),
            1
        );

        // Recover coordinator from disk: spent budget from crashed attempt remains debited (non-refundable)!
        let resume_state = crate::artifacts::read_verified_round_resume_state(&run_dir)
            .unwrap()
            .unwrap();
        #[allow(deprecated)]
        let mut recovered = SwarmRoundCoordinator::<LinearModel>::from_resume_state(
            config,
            linear_model_2_1(&[0.0, 0.0, 0.0]),
            &resume_state,
            &session,
        )
        .unwrap();
        recovered
            .configure_dp_accountant(&run_dir, dp_profile)
            .unwrap();
        assert_eq!(recovered.dp_accountant().unwrap().committed_rounds(), 1);

        // Further spending eventually hits PrivacyBudgetExhausted BEFORE any checkpoint writes
        for r in 2..=20 {
            let committed_before = recovered.dp_accountant().unwrap().committed_rounds();
            if let Err(RoundError::PrivacyBudget(CryptoError::PrivacyBudgetExhausted)) = recovered
                .pre_release_dp_debit_if_enabled(&mut session, r, 2_000 + r, None, Some(1.0))
            {
                assert_eq!(
                    recovered.dp_accountant().unwrap().committed_rounds(),
                    committed_before
                );
                let _ = fs::remove_dir_all(&base);
                return;
            }
        }
        panic!("expected PrivacyBudgetExhausted within budget ceiling");
    }

    #[test]
    fn lora_and_flora_cohort_aggregation_enforces_auth_quorum_and_dp_budget() {
        use swarm_torch_core::aggregation::FedAvg;
        use swarm_torch_core::dp::{
            DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1,
            PrivacyAccountingClassificationV1,
        };
        use swarm_torch_models::lora::{
            LoraAdapterPairV1, LoraAggregationModeV1, LoraAggregationOutcomeV1,
        };
        use swarm_torch_models::safetensors_v1::LoRaAdapterDescriptorV1;

        let base = temp_dir("lora_coord");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x99; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let p1 = TestPeer::new(1);
        let p2 = TestPeer::new(2);
        let p_intruder = TestPeer::new(9);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::default(),
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config,
            linear_model_2_1(&[0.0, 1.0, 1.0]),
            &mut session,
            1_000,
        )
        .unwrap();
        coord
            .configure_dp_accountant(
                &run_dir,
                DpMechanismProfileV1 {
                    unit: DpUnitV1::ClientLevel,
                    adjacency: DpAdjacencyV1::AddRemoveOne,
                    clipping_radius_tau: 1.0,
                    noise_multiplier: 2.0,
                    target_delta: 1e-5,
                    max_epsilon: 10.0,
                    conversion_formula: DpConversionFormulaV1::Balle2020Tight,
                    classification:
                        PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
                },
            )
            .unwrap();
        let s1 = p1.establish_session(&mut coord, 1, 0xF1, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0xF2, 100);

        let pair1 = LoraAdapterPairV1::try_new(
            LoRaAdapterDescriptorV1 {
                target_module: "q_proj".to_string(),
                tensor_a_name: "q_proj.lora_A".to_string(),
                tensor_b_name: "q_proj.lora_B".to_string(),
                rank: 1,
                alpha: 1.0,
            },
            2,
            2,
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        )
        .unwrap();
        let pair2 = LoraAdapterPairV1::try_new(
            LoRaAdapterDescriptorV1 {
                target_module: "q_proj".to_string(),
                tensor_a_name: "q_proj.lora_A".to_string(),
                tensor_b_name: "q_proj.lora_B".to_string(),
                rank: 2,
                alpha: 2.0,
            },
            2,
            2,
            vec![0.0, 1.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
        )
        .unwrap();

        // Intruder peer fails Stage A replay/allowlist check BEFORE DP debit or Stage C!
        let intruder_err =
            verify_lora_delivery(&mut coord, &p_intruder, [0x99; 32], 1, 1, 100, &pair2);
        assert!(matches!(
            intruder_err,
            Err(RoundError::ReplayRejected(
                ReplayError::UnauthorizedPeer { .. }
            ))
        ));
        assert_eq!(coord.dp_accountant().unwrap().committed_rounds(), 0);

        // Authorized heterogeneous-rank FLoRA aggregation merges exact base weight and debits DP ledger
        let v1 = verify_lora_delivery(&mut coord, &p1, s1, 1, 1, 100, &pair1).unwrap();
        let v2 = verify_lora_delivery(&mut coord, &p2, s2, 1, 1, 100, &pair2).unwrap();
        let mut base_w = vec![0.0f32; 4];
        let outcome = coord
            .aggregate_lora_cohort_verified_round(
                &[v1, v2],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 16,
                },
                &FedAvg,
                Some(&mut base_w),
                1.0,
                &mut session,
                2_100,
            )
            .unwrap();
        assert!(matches!(
            outcome,
            LoraAggregationOutcomeV1::FloraStacked { .. }
        ));
        assert_eq!(coord.dp_accountant().unwrap().committed_rounds(), 1);
        assert!(base_w.iter().any(|&v| v != 0.0));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn certified_dp_fails_closed_and_lora_preserves_state_on_invalid_merge_or_dp_exhaustion() {
        use swarm_torch_core::aggregation::FedAvg;
        use swarm_torch_core::dp::{
            DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1,
            PrivacyAccountingClassificationV1,
        };
        use swarm_torch_models::lora::{LoraAdapterPairV1, LoraAggregationModeV1};
        use swarm_torch_models::safetensors_v1::LoRaAdapterDescriptorV1;

        let base = temp_dir("st02_lora_atomicity");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x9a; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let p1 = TestPeer::new(1);
        let p2 = TestPeer::new(2);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::default(),
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config,
            linear_model_2_1(&[0.0, 1.0, 1.0]),
            &mut session,
            1_000,
        )
        .unwrap();
        assert!(!coord.supports_certified_differential_privacy());

        // Certified DP mode MUST fail closed in v0.2 before configuring any accountant.
        let certified_profile = DpMechanismProfileV1 {
            unit: DpUnitV1::ClientLevel,
            adjacency: DpAdjacencyV1::AddRemoveOne,
            clipping_radius_tau: 1.0,
            noise_multiplier: 1.5,
            target_delta: 1e-5,
            max_epsilon: 5.0,
            conversion_formula: DpConversionFormulaV1::Balle2020Tight,
            classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
        };
        assert!(matches!(
            coord.configure_dp_accountant(&run_dir, certified_profile),
            Err(RoundError::Config(
                RoundConfigError::UnsupportedCertifiedDifferentialPrivacy
            ))
        ));
        assert!(coord.dp_accountant().is_none());

        // Configure experimental hypothetical accountant with tight max_epsilon that allows 1 round but rejects round 2.
        coord
            .configure_dp_accountant(
                &run_dir,
                DpMechanismProfileV1 {
                    unit: DpUnitV1::ClientLevel,
                    adjacency: DpAdjacencyV1::AddRemoveOne,
                    clipping_radius_tau: 1.0,
                    noise_multiplier: 1.5,
                    target_delta: 1e-5,
                    max_epsilon: 3.0,
                    conversion_formula: DpConversionFormulaV1::Balle2020Tight,
                    classification:
                        PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
                },
            )
            .unwrap();
        assert!(!coord
            .dp_accountant()
            .unwrap()
            .guarantees_differential_privacy());
        let s1 = p1.establish_session(&mut coord, 1, 0xA1, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0xA2, 100);

        let pair1 = LoraAdapterPairV1::try_new(
            LoRaAdapterDescriptorV1 {
                target_module: "q_proj".to_string(),
                tensor_a_name: "q_proj.lora_A".to_string(),
                tensor_b_name: "q_proj.lora_B".to_string(),
                rank: 1,
                alpha: 1.0,
            },
            2,
            2,
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        )
        .unwrap();
        let pair2 = LoraAdapterPairV1::try_new(
            LoRaAdapterDescriptorV1 {
                target_module: "q_proj".to_string(),
                tensor_a_name: "q_proj.lora_A".to_string(),
                tensor_b_name: "q_proj.lora_B".to_string(),
                rank: 1,
                alpha: 1.0,
            },
            2,
            2,
            vec![0.0, 1.0],
            vec![0.0, 2.0],
        )
        .unwrap();

        // 1. Mismatched `base_weights_to_merge` length MUST fail BEFORE pre_release_dp_debit_if_enabled!
        let v1_step1 = verify_lora_delivery(&mut coord, &p1, s1, 1, 1, 100, &pair1).unwrap();
        let v2_step1 = verify_lora_delivery(&mut coord, &p2, s2, 1, 1, 100, &pair2).unwrap();
        let mut wrong_len_base = vec![1.25f32; 3];
        let wrong_len_err = coord.aggregate_lora_cohort_verified_round(
            &[v1_step1, v2_step1],
            LoraAggregationModeV1::FloraStackingV1 {
                max_stacked_rank: 8,
            },
            &FedAvg,
            Some(&mut wrong_len_base),
            1.0,
            &mut session,
            2_000,
        );
        assert!(wrong_len_err.is_err());
        assert_eq!(wrong_len_base, vec![1.25f32; 3]);
        assert_eq!(coord.dp_accountant().unwrap().committed_rounds(), 0);
        assert!(crate::artifacts::read_dp_ledger_entries(&run_dir)
            .unwrap()
            .is_empty());

        // 2. First valid LoRA round consumes budget cleanly (committed_rounds = 1).
        let v1_step2 = verify_lora_delivery(&mut coord, &p1, s1, 1, 2, 100, &pair1).unwrap();
        let v2_step2 = verify_lora_delivery(&mut coord, &p2, s2, 1, 2, 100, &pair2).unwrap();
        let mut base_w = vec![0.5f32; 4];
        coord
            .aggregate_lora_cohort_verified_round(
                &[v1_step2, v2_step2],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 8,
                },
                &FedAvg,
                Some(&mut base_w),
                1.0,
                &mut session,
                2_100,
            )
            .unwrap();
        assert_eq!(coord.dp_accountant().unwrap().committed_rounds(), 1);
        let after_first = base_w.clone();

        // 3. Budget exhaustion on subsequent LoRA round MUST fail WITHOUT mutating `base_w`!
        let mut hit_budget_err = false;
        for step in 0..8u64 {
            let seq = 3 + step;
            let v1 = verify_lora_delivery(&mut coord, &p1, s1, 1, seq, 100, &pair1).unwrap();
            let v2 = verify_lora_delivery(&mut coord, &p2, s2, 1, seq, 100, &pair2).unwrap();
            let mut candidate_w = after_first.clone();
            match coord.aggregate_lora_cohort_verified_round(
                &[v1, v2],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 8,
                },
                &FedAvg,
                Some(&mut candidate_w),
                1.0,
                &mut session,
                2_200 + step,
            ) {
                Ok(_) => {}
                Err(RoundError::PrivacyBudget(_)) => {
                    assert_eq!(
                        candidate_w, after_first,
                        "DP budget failure must leave base_weights_to_merge untouched"
                    );
                    hit_budget_err = true;
                    break;
                }
                Err(other) => panic!("unexpected error on exhaustion step: {other}"),
            }
        }
        assert!(hit_budget_err, "expected PrivacyBudget exhaustion error");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn stage_ab_failures_leave_replay_coordinator_and_model_untouched() {
        let base = temp_dir("st01_zero_mutation");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x44; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(11);
        let p2 = TestPeer::new(12);
        let intruder = TestPeer::new(99);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config,
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut session,
            1_000,
        )
        .unwrap();

        // 0. Delivery BEFORE handshake MUST fail with RecoveryRequired without consuming state.
        let pre_hs_delivery = p1.build_dense_delivery(
            &coord,
            [0x11; 32],
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        assert!(matches!(
            coord.submit_verified_delivery(
                &pre_hs_delivery,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                100
            ),
            Err(RoundError::ReplayRejected(
                ReplayError::RecoveryRequired { .. }
            ))
        ));
        assert_eq!(coord.replay_cache_size(), 0);

        let s1 = p1.establish_session(&mut coord, 1, 0x55, 100);
        assert_eq!(coord.replay_cache_size(), 1);
        let gen_before = coord.coordinator_generation();

        // 1. Corrupt signature in Stage A -> AuthenticationFailed, ZERO state change
        let valid_struct = BundledGossipDeliveryV2::new_signed(
            &p1.auth,
            InnerUpdateV2 {
                originator_public_key: *p1.auth.key_pair().public_key(),
                swarm_instance_id: coord.swarm_instance_id(),
                run_id: *run_id.as_bytes(),
                round_id: 1,
                contribution_id: 1,
                base_checkpoint_digest: *coord.parent_digest().as_bytes(),
                context_policy_digest: coord.context_policy_digest(),
                payload_kind: PayloadKindV1::ModelDeltaV1,
                compression_schema_version: COMPRESSION_SCHEMA_DENSE_F32_LE_V1,
                num_elements: 3,
                reported_examples: 0,
                encoded_payload: encode_dense_update_payload_v2(&[0.5, 0.5, 0.5]).unwrap(),
                originator_signature: Vec::new(),
            },
            coord.coordinator_peer_id(),
            s1,
            1,
            1,
            100,
        );
        let mut bad_sig = valid_struct.clone();
        bad_sig.inner.originator_signature[0] ^= 0xFF;
        let bad_sig_bytes = bad_sig.to_bytes().unwrap();
        assert!(matches!(
            coord.submit_verified_delivery(&bad_sig_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100),
            Err(RoundError::AuthenticationFailed(_))
        ));

        // 2. Wrong context_policy_digest (valid signature by p1, seq=1) -> fails Stage B WITHOUT consuming seq=1!
        let mut wrong_policy_inner = valid_struct.inner.clone();
        wrong_policy_inner.context_policy_digest[0] ^= 0xAA;
        let wrong_policy_bytes = BundledGossipDeliveryV2::new_signed(
            &p1.auth,
            wrong_policy_inner,
            coord.coordinator_peer_id(),
            s1,
            1,
            1,
            100,
        )
        .to_bytes()
        .unwrap();
        assert!(matches!(
            coord.submit_verified_delivery(
                &wrong_policy_bytes,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                100
            ),
            Err(RoundError::ContextPolicyDigestMismatch { .. })
        ));

        // 3. Wrong base_checkpoint_digest -> BaseCheckpointMismatch, seq=1 still unused!
        let wrong_base_bytes = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            1,
            CheckpointContentDigestV1([0x99; 32]),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        assert!(matches!(
            coord.submit_verified_delivery(
                &wrong_base_bytes,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                100
            ),
            Err(RoundError::BaseCheckpointMismatch { .. })
        ));

        // 4. Wrong parameter dimension -> DimensionMismatch, seq=1 still unused!
        let wrong_dim_bytes = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1],
            0,
        );
        assert!(matches!(
            coord.submit_verified_delivery(
                &wrong_dim_bytes,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                100
            ),
            Err(RoundError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));

        // 5. Non-finite payload -> NonFiniteInput in Stage A, seq=1 still unused!
        let nan_bytes = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, f32::NAN, 0.1],
            0,
        );
        assert!(matches!(
            coord.submit_verified_delivery(&nan_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100),
            Err(RoundError::NonFiniteInput { coord_index: 1, .. })
        ));

        // 6. Unauthorized intruder -> UnauthorizedPeer, zero mutation
        let intruder_bytes = intruder.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        assert!(matches!(
            coord.submit_verified_delivery(&intruder_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100),
            Err(RoundError::ReplayRejected(
                ReplayError::UnauthorizedPeer { .. }
            ))
        ));

        assert_eq!(coord.admitted_count(), 0);
        assert_eq!(coord.equivocated_count(), 0);
        assert_eq!(coord.replay_cache_size(), 1);
        assert_eq!(coord.coordinator_generation(), gen_before);

        // Proof that seq=1 was NOT burned by any of errors 1..6: valid p1 submission with seq=1 succeeds!
        let valid_bytes = valid_struct.to_bytes().unwrap();
        assert!(coord
            .submit_verified_delivery(&valid_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .is_ok());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    #[allow(deprecated)]
    fn stale_verified_submission_and_deprecated_entrypoints_fail_closed() {
        use swarm_torch_core::aggregation::FedAvg;
        use swarm_torch_models::lora::{LoraAdapterPairV1, LoraAggregationModeV1};
        use swarm_torch_models::safetensors_v1::LoRaAdapterDescriptorV1;

        let base = temp_dir("st01_stale_gen_and_legacy");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x55; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(21);
        let p2 = TestPeer::new(22);
        let p3 = TestPeer::new(23);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id, p3.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config,
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut session,
            1_000,
        )
        .unwrap();

        // All 3 deprecated unverified entrypoints fail closed with UnverifiedSubmissionRejected
        let legacy_sub = PeerRoundSubmissionV1 {
            run_id,
            round_id: 1,
            base_checkpoint_digest: coord.parent_digest(),
            sender: p1.peer_id,
            sequence: 1,
            timestamp_secs: 100,
            update_kind: UpdateKindV1::default(),
            values: vec![0.1, 0.1, 0.1],
            reported_examples: None,
        };
        assert!(matches!(
            coord.submit_peer_update(legacy_sub.clone(), 100),
            Err(RoundError::UnverifiedSubmissionRejected)
        ));
        assert!(matches!(
            coord.step_fedasync_stale_submission(legacy_sub, 100, 1.0, &mut session, 1_100),
            Err(RoundError::UnverifiedSubmissionRejected)
        ));
        let dummy_pair = LoraAdapterPairV1::try_new(
            LoRaAdapterDescriptorV1 {
                target_module: "q".to_string(),
                tensor_a_name: "a".to_string(),
                tensor_b_name: "b".to_string(),
                rank: 1,
                alpha: 1.0,
            },
            1,
            1,
            vec![1.0],
            vec![1.0],
        )
        .unwrap();
        assert!(matches!(
            coord.aggregate_lora_cohort_round(
                &[(p1.peer_id, dummy_pair.clone()), (p2.peer_id, dummy_pair)],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 4
                },
                &FedAvg,
                None,
                1.0,
                &mut session,
                1_200,
            ),
            Err(RoundError::UnverifiedSubmissionRejected)
        ));
        assert_eq!(coord.admitted_count(), 0);

        // Mint a VerifiedRoundSubmissionV1 for p3 at coordinator_generation=0, then finalize round 1 via p1 + p2.
        let s1 = p1.establish_session(&mut coord, 1, 0x01, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0x02, 100);
        let s3 = p3.establish_session(&mut coord, 1, 0x03, 100);

        let b3 = p3.build_dense_delivery(
            &coord,
            s3,
            1,
            1,
            100,
            1,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.5, 0.5, 0.5],
            0,
        );
        let prep3 = coord
            .prepare_verified_delivery(&b3, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap();
        let val3 = coord
            .expected_round_context()
            .validate_sync_submission(prep3)
            .unwrap();
        let verified_p3_gen0 = coord.commit_prepared_submission(val3).unwrap();

        for (p, s) in [(&p1, s1), (&p2, s2)] {
            let bytes = p.build_dense_delivery(
                &coord,
                s,
                1,
                1,
                100,
                1,
                coord.parent_digest(),
                PayloadKindV1::ModelDeltaV1,
                &[0.1, 0.1, 0.1],
                0,
            );
            coord
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap();
        }
        let _ = coord
            .finalize_round::<SgdOptimizer<LinearModel>, DummyEvaluator>(
                &[],
                None,
                None,
                &mut session,
                2_000,
            )
            .unwrap();
        assert_eq!(coord.coordinator_generation(), 1);

        // Presenting the old VerifiedRoundSubmissionV1 (generation 0) to generation 1 MUST fail!
        let stale_gen_err = coord.submit_verified_update(verified_p3_gen0).unwrap_err();
        assert!(matches!(
            stale_gen_err,
            RoundError::StaleCoordinatorGeneration {
                expected: 1,
                got: 0
            }
        ));
        assert_eq!(coord.admitted_count(), 0);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resumed_coordinator_enforces_fresh_v2_handshake_and_timestamp_floor() {
        let base = temp_dir("st03_timestamp_floor");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x67; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(31);
        let p2 = TestPeer::new(32);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let t_r1_secs = 1_700_000_010u64;
        let t_r1_nanos = t_r1_secs * 1_000_000_000;
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config.clone(),
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut session,
            t_r1_nanos - 1_000_000_000,
        )
        .unwrap();

        let s1 = p1.establish_session(&mut coord, 1, 0x11, t_r1_secs as u32);
        let s2 = p2.establish_session(&mut coord, 1, 0x22, t_r1_secs as u32);
        for (p, s) in [(&p1, s1), (&p2, s2)] {
            let bytes = p.build_dense_delivery(
                &coord,
                s,
                1,
                1,
                t_r1_secs as u32,
                1,
                coord.parent_digest(),
                PayloadKindV1::ModelDeltaV1,
                &[0.1, 0.1, 0.1],
                0,
            );
            coord
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, t_r1_secs)
                .unwrap();
        }
        let _ = coord
            .finalize_round::<SgdOptimizer<LinearModel>, DummyEvaluator>(
                &[],
                None,
                None,
                &mut session,
                t_r1_nanos,
            )
            .unwrap();

        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let _ = session.commit_and_close(None).unwrap();
        let (resumed_session, resume_state) = DataOpsSession::resume(
            &run_dir,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let resume_state = resume_state.unwrap();

        #[allow(deprecated)]
        let mut resumed: SwarmRoundCoordinator<LinearModel> =
            SwarmRoundCoordinator::from_resume_state(
                config,
                LinearModel::new(2, 1).unwrap(),
                &resume_state,
                &resumed_session,
            )
            .unwrap();

        // Pre-floor session challenge/handshake or delivery is rejected after resume!
        let s1_r2 = p1.establish_session(&mut resumed, 2, 0x33, (t_r1_secs + 1) as u32);
        let below_floor_bytes = p1.build_dense_delivery(
            &resumed,
            s1_r2,
            2,
            1,
            (t_r1_secs - 1) as u32,
            2,
            resumed.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        let floor_err = resumed
            .submit_verified_delivery(
                &below_floor_bytes,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                t_r1_secs + 1,
            )
            .unwrap_err();
        assert!(matches!(
            floor_err,
            RoundError::ReplayRejected(ReplayError::TimestampBelowFloor { .. })
        ));

        // Delivery at or above floor succeeds!
        let at_floor_bytes = p1.build_dense_delivery(
            &resumed,
            s1_r2,
            2,
            1,
            (t_r1_secs + 1) as u32,
            2,
            resumed.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        assert!(resumed
            .submit_verified_delivery(
                &at_floor_bytes,
                DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
                t_r1_secs + 1
            )
            .is_ok());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st04_finalize_round_failure_in_optimizer_or_fitness_leaves_state_untouched() {
        use swarm_torch_models::optim::SgdOptimizer;

        struct FailingEvaluator;
        impl FitnessFunction for FailingEvaluator {
            type Model = LinearModel;
            type Error = &'static str;
            fn evaluate(&self, _model: &Self::Model) -> Result<f32, Self::Error> {
                Err("injected fitness failure")
            }
        }

        let base = temp_dir("st04_failure_atomicity");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x61; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(41);
        let p2 = TestPeer::new(42);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::GradientV1,
            aggregation: RobustAggregation::CenteredClipping {
                tau: 10.0,
                max_iterations: 2,
            },
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config,
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut session,
            1_000,
        )
        .unwrap();

        let mut opt = SgdOptimizer::<LinearModel, 16>::with_hyperparams(0.1, 0.9, 0.0).unwrap();

        // Commit Round 1 cleanly so optimizer velocity, cclip_center, and committed_head advance.
        let s1 = p1.establish_session(&mut coord, 1, 0x01, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0x02, 100);
        for (p, s) in [(&p1, s1), (&p2, s2)] {
            let b = p.build_dense_delivery(
                &coord,
                s,
                1,
                1,
                100,
                1,
                coord.parent_digest(),
                PayloadKindV1::GradientV1,
                &[0.5, -0.25, 0.1],
                0,
            );
            coord
                .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap();
        }
        coord
            .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                &[],
                Some(&mut opt),
                None,
                &mut session,
                2_000,
            )
            .unwrap();

        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let head_before = crate::artifacts::read_committed_head_v1(&run_dir).unwrap();
        let model_before = coord.model().parameters().to_vec();
        let vel_before = opt.velocity().to_vec();
        let cclip_before = coord.cclip_center().unwrap().to_vec();
        let gen_before = coord.coordinator_generation();

        // Round 2: admit valid gradients, then fail inside `FailingEvaluator` -> ZERO mutations!
        for (p, s) in [(&p1, s1), (&p2, s2)] {
            let b = p.build_dense_delivery(
                &coord,
                s,
                1,
                2,
                110,
                2,
                coord.parent_digest(),
                PayloadKindV1::GradientV1,
                &[0.4, 0.4, -0.2],
                0,
            );
            coord
                .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 110)
                .unwrap();
        }
        let err = coord
            .finalize_round(
                &[],
                Some(&mut opt),
                Some(&FailingEvaluator),
                &mut session,
                3_000,
            )
            .unwrap_err();
        assert!(matches!(err, RoundError::FitnessEvaluation(_)));
        assert_eq!(coord.current_round(), 2);
        assert_eq!(coord.coordinator_generation(), gen_before);
        assert_eq!(coord.model().parameters(), &model_before[..]);
        assert_eq!(opt.velocity(), &vel_before[..]);
        assert_eq!(coord.cclip_center().unwrap(), &cclip_before[..]);
        assert_eq!(
            crate::artifacts::read_committed_head_v1(&run_dir).unwrap(),
            head_before
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st04_crash_seams_before_and_after_committed_head_preserve_strict_recovery() {
        use swarm_torch_core::crypto::KeyPair;
        use swarm_torch_models::optim::SgdOptimizer;

        let base = temp_dir("st04_crash_seams");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x62; 16]);
        let signer = KeyPair::from_seed([0x88; 32]).unwrap();
        let strict_policy = TrustPolicy::Strict {
            trusted_public_keys: vec![*signer.public_key()],
            expected_run_id: Some(run_id),
            min_signed_unix_nanos: None,
        };

        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let p1 = TestPeer::new(43);
        let p2 = TestPeer::new(44);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::GradientV1,
            aggregation: RobustAggregation::CenteredClipping {
                tau: 10.0,
                max_iterations: 2,
            },
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config.clone(),
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut session,
            1_000,
        )
        .unwrap()
        .with_manifest_signer(signer.clone());
        let mut opt = SgdOptimizer::<LinearModel, 16>::with_hyperparams(0.05, 0.9, 0.0).unwrap();

        // Commit Round 1 cleanly (signed manifest + committed_head gen=1).
        let s1 = p1.establish_session(&mut coord, 1, 0x01, 100);
        let s2 = p2.establish_session(&mut coord, 1, 0x02, 100);
        for (p, s) in [(&p1, s1), (&p2, s2)] {
            let b = p.build_dense_delivery(
                &coord,
                s,
                1,
                1,
                100,
                1,
                coord.parent_digest(),
                PayloadKindV1::GradientV1,
                &[0.2, -0.1, 0.3],
                0,
            );
            coord
                .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
                .unwrap();
        }
        coord
            .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                &[],
                Some(&mut opt),
                None,
                &mut session,
                2_000,
            )
            .unwrap();
        let r1_model = coord.model().parameters().to_vec();
        let r1_vel = opt.velocity().to_vec();
        let r1_cclip = coord.cclip_center().unwrap().to_vec();
        let r1_gen = coord.coordinator_generation();
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let r1_head = crate::artifacts::read_committed_head_v1(&run_dir).unwrap();
        assert_eq!(r1_head.generation, r1_gen);
        assert_eq!(r1_head.round_id, 1);

        // Round 2 Case A: Inject crash AFTER manifest written/signed BEFORE committed_head swap!
        for (p, s) in [(&p1, s1), (&p2, s2)] {
            let b = p.build_dense_delivery(
                &coord,
                s,
                1,
                2,
                110,
                2,
                coord.parent_digest(),
                PayloadKindV1::GradientV1,
                &[0.3, 0.1, -0.2],
                0,
            );
            coord
                .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 110)
                .unwrap();
        }
        coord.set_test_commit_fail_point(CommitFailPoint::AfterManifestBeforeCommittedHead);
        let crash_before_head = coord
            .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                &[],
                Some(&mut opt),
                None,
                &mut session,
                3_000,
            )
            .unwrap_err();
        assert!(matches!(crash_before_head, RoundError::Io(_)));

        // In-memory state AND committed_head still point to Round 1!
        assert_eq!(coord.current_round(), 2);
        assert_eq!(coord.model().parameters(), &r1_model[..]);
        assert_eq!(opt.velocity(), &r1_vel[..]);
        assert_eq!(
            crate::artifacts::read_committed_head_v1(&run_dir).unwrap(),
            r1_head
        );

        // Simulated restart right now under TrustPolicy::Strict MUST recover Round 1 cleanly!
        drop(session);
        let (restarted_session, _) = DataOpsSession::resume(
            &run_dir,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let mut recovered_opt = SgdOptimizer::<LinearModel, 16>::new(0.99).unwrap();
        let mut recovered = SwarmRoundCoordinator::resume_exact_training_state(
            config.clone(),
            LinearModel::new(2, 1).unwrap(),
            Some(&mut recovered_opt),
            &restarted_session,
            &strict_policy,
        )
        .unwrap()
        .with_manifest_signer(signer);
        assert_eq!(recovered.current_round(), 2);
        assert_eq!(recovered.coordinator_generation(), r1_gen);
        assert_eq!(recovered.model().parameters(), &r1_model[..]);
        assert_eq!(recovered_opt.velocity(), &r1_vel[..]);
        assert_eq!(recovered.cclip_center().unwrap(), &r1_cclip[..]);

        // Round 2 Case B: Inject crash AFTER CAS (before manifest) on recovered coordinator, then
        // inject crash AFTER committed_head BEFORE memory -> committed_head advances to Round 2!
        let s1_r2 = p1.establish_session(&mut recovered, 2, 0x11, 120);
        let s2_r2 = p2.establish_session(&mut recovered, 2, 0x12, 120);
        for (p, s) in [(&p1, s1_r2), (&p2, s2_r2)] {
            let b = p.build_dense_delivery(
                &recovered,
                s,
                2,
                1,
                120,
                2,
                recovered.parent_digest(),
                PayloadKindV1::GradientV1,
                &[0.3, 0.1, -0.2],
                0,
            );
            recovered
                .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 120)
                .unwrap();
        }
        let mut sess2 = restarted_session;
        recovered.set_test_commit_fail_point(CommitFailPoint::AfterCasBeforeManifest);
        assert!(recovered
            .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                &[],
                Some(&mut recovered_opt),
                None,
                &mut sess2,
                4_000,
            )
            .is_err());
        assert_eq!(recovered.current_round(), 2);

        recovered.set_test_commit_fail_point(CommitFailPoint::AfterCommittedHeadBeforeMemory);
        assert!(recovered
            .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                &[],
                Some(&mut recovered_opt),
                None,
                &mut sess2,
                4_000,
            )
            .is_err());
        // Process crashed before Step 5 memory update, but disk committed_head already switched to Round 2!
        assert_eq!(recovered.current_round(), 2);
        let head_after = crate::artifacts::read_committed_head_v1(&run_dir).unwrap();
        assert_eq!(head_after.round_id, 2);

        drop(sess2);
        let (sess3, _) = DataOpsSession::resume(
            &run_dir,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let mut opt_r2 = SgdOptimizer::<LinearModel, 16>::new(0.99).unwrap();
        let rec_r2 = SwarmRoundCoordinator::resume_exact_training_state(
            config,
            LinearModel::new(2, 1).unwrap(),
            Some(&mut opt_r2),
            &sess3,
            &strict_policy,
        )
        .unwrap();
        assert_eq!(rec_r2.current_round(), 3);
        assert_eq!(rec_r2.coordinator_generation(), r1_gen + 1);
        assert_ne!(rec_r2.model().parameters(), &r1_model[..]);
        assert_ne!(opt_r2.velocity(), &r1_vel[..]);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st04_exact_resume_matches_continuous_run_and_weights_only_clears_lineage() {
        use swarm_torch_core::aggregation::FedAsyncStalenessDescriptorV1;
        use swarm_torch_core::crypto::KeyPair;
        use swarm_torch_models::optim::SgdOptimizer;

        let base_cont = temp_dir("st04_continuous");
        let base_res = temp_dir("st04_resumed");
        let _ = fs::remove_dir_all(&base_cont);
        let _ = fs::remove_dir_all(&base_res);
        fs::create_dir_all(&base_cont).unwrap();
        fs::create_dir_all(&base_res).unwrap();

        let run_id = RunId::from_bytes([0x63; 16]);
        let signer = KeyPair::from_seed([0x7A; 32]).unwrap();
        let strict_policy = TrustPolicy::Strict {
            trusted_public_keys: vec![*signer.public_key()],
            expected_run_id: Some(run_id),
            min_signed_unix_nanos: None,
        };
        let p1 = TestPeer::new(45);
        let p2 = TestPeer::new(46);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::GradientV1,
            aggregation: RobustAggregation::CenteredClipping {
                tau: 10.0,
                max_iterations: 2,
            },
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id, p2.peer_id].into_iter().collect(),
            min_quorum: 2,
            max_staleness: 2,
            max_clock_skew_secs: 60,
        };
        let fedasync = FedAsyncStalenessDescriptorV1 {
            base_alpha: 0.4,
            max_staleness_rounds: 2,
            polynomial_decay_exponent: 0.5,
        };

        // Run A: Continuous 2 rounds with GradientV1 + CenteredClipping + FedAsync enabled.
        let mut sess_a = DataOpsSession::create(
            &base_cont,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let mut coord_a = SwarmRoundCoordinator::initialize(
            run_id,
            config.clone(),
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut sess_a,
            1_000,
        )
        .unwrap();
        coord_a.enable_fedasync(fedasync, 0, None).unwrap();
        let mut opt_a = SgdOptimizer::<LinearModel, 16>::with_hyperparams(0.05, 0.9, 1e-4).unwrap();
        let s1_a = p1.establish_session(&mut coord_a, 1, 0x01, 100);
        let s2_a = p2.establish_session(&mut coord_a, 1, 0x02, 100);
        for r in [1u64, 2] {
            for (p, s) in [(&p1, s1_a), (&p2, s2_a)] {
                let b = p.build_dense_delivery(
                    &coord_a,
                    s,
                    1,
                    r,
                    100 + r as u32,
                    r,
                    coord_a.parent_digest(),
                    PayloadKindV1::GradientV1,
                    &[0.1 * r as f32, -0.2, 0.05],
                    0,
                );
                coord_a
                    .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100 + r)
                    .unwrap();
            }
            coord_a
                .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                    &[],
                    Some(&mut opt_a),
                    None,
                    &mut sess_a,
                    (100 + r) * 1_000_000_000,
                )
                .unwrap();
        }

        // Run B: Same Round 1, then sign/close, ExactTrainingResume, then Round 2!
        let mut sess_b1 = DataOpsSession::create(
            &base_res,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let mut coord_b1 = SwarmRoundCoordinator::initialize(
            run_id,
            config.clone(),
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut sess_b1,
            1_000,
        )
        .unwrap();
        coord_b1.enable_fedasync(fedasync, 0, None).unwrap();
        let mut opt_b1 =
            SgdOptimizer::<LinearModel, 16>::with_hyperparams(0.05, 0.9, 1e-4).unwrap();
        let s1_b1 = p1.establish_session(&mut coord_b1, 1, 0x01, 100);
        let s2_b1 = p2.establish_session(&mut coord_b1, 1, 0x02, 100);
        for (p, s) in [(&p1, s1_b1), (&p2, s2_b1)] {
            let b = p.build_dense_delivery(
                &coord_b1,
                s,
                1,
                1,
                101,
                1,
                coord_b1.parent_digest(),
                PayloadKindV1::GradientV1,
                &[0.1, -0.2, 0.05],
                0,
            );
            coord_b1
                .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 101)
                .unwrap();
        }
        coord_b1
            .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                &[],
                Some(&mut opt_b1),
                None,
                &mut sess_b1,
                101 * 1_000_000_000,
            )
            .unwrap();
        let run_dir_b = sess_b1.sink().bundle().run_dir().to_path_buf();
        sess_b1.commit_and_close(Some(&signer)).unwrap();

        let (mut sess_b2, _) = DataOpsSession::resume(
            &run_dir_b,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let mut opt_b2 = SgdOptimizer::<LinearModel, 16>::new(0.5).unwrap();
        let mut coord_b2 = SwarmRoundCoordinator::resume_exact_training_state(
            config.clone(),
            LinearModel::new(2, 1).unwrap(),
            Some(&mut opt_b2),
            &sess_b2,
            &strict_policy,
        )
        .unwrap();
        // Round 0 historical anchor from pre-crash is still retained!
        assert!(coord_b2.historical_anchor_weights(0).is_some());
        assert!(coord_b2.cclip_center().is_some());

        let s1_b2 = p1.establish_session(&mut coord_b2, 2, 0x21, 102);
        let s2_b2 = p2.establish_session(&mut coord_b2, 2, 0x22, 102);
        for (p, s) in [(&p1, s1_b2), (&p2, s2_b2)] {
            let b = p.build_dense_delivery(
                &coord_b2,
                s,
                2,
                1,
                102,
                2,
                coord_b2.parent_digest(),
                PayloadKindV1::GradientV1,
                &[0.2, -0.2, 0.05],
                0,
            );
            coord_b2
                .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 102)
                .unwrap();
        }
        coord_b2
            .finalize_round::<SgdOptimizer<LinearModel, 16>, DummyEvaluator>(
                &[],
                Some(&mut opt_b2),
                None,
                &mut sess_b2,
                102 * 1_000_000_000,
            )
            .unwrap();

        // Resumed Run B matches continuous Run A bit-for-bit!
        assert_eq!(coord_b2.model().parameters(), coord_a.model().parameters());
        assert_eq!(opt_b2.velocity(), opt_a.velocity());
        assert_eq!(coord_b2.cclip_center(), coord_a.cclip_center());
        assert_eq!(
            coord_b2.coordinator_generation(),
            coord_a.coordinator_generation()
        );

        // Negative ExactTrainingResume tests: untrusted key, None optimizer, changed config fail closed!
        let bad_signer = KeyPair::from_seed([0x99; 32]).unwrap();
        assert!(SwarmRoundCoordinator::resume_exact_training_state(
            config.clone(),
            LinearModel::new(2, 1).unwrap(),
            Some(&mut opt_b2),
            &sess_b2,
            &TrustPolicy::Strict {
                trusted_public_keys: vec![*bad_signer.public_key()],
                expected_run_id: Some(run_id),
                min_signed_unix_nanos: None,
            },
        )
        .is_err());
        assert!(
            SwarmRoundCoordinator::resume_exact_training_state::<SgdOptimizer<LinearModel, 16>>(
                config.clone(),
                LinearModel::new(2, 1).unwrap(),
                None,
                &sess_b2,
                &TrustPolicy::LegacyPermissive,
            )
            .is_err()
        );
        let mut wrong_cfg = config.clone();
        wrong_cfg.max_staleness = 0;
        assert!(SwarmRoundCoordinator::resume_exact_training_state(
            wrong_cfg,
            LinearModel::new(2, 1).unwrap(),
            Some(&mut opt_b2),
            &sess_b2,
            &TrustPolicy::LegacyPermissive,
        )
        .is_err());

        // WeightsOnlyRestore: clears cclip_center & round 0 anchor, preserves weights cleanly!
        sess_b2.commit_and_close(Some(&signer)).unwrap();
        let (sess_b3, _) = DataOpsSession::resume(
            &run_dir_b,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let weights_only = SwarmRoundCoordinator::restore_weights_only(
            config,
            LinearModel::new(2, 1).unwrap(),
            &sess_b3,
            &strict_policy,
        )
        .unwrap();
        assert_eq!(
            weights_only.model().parameters(),
            coord_b2.model().parameters()
        );
        assert!(weights_only.cclip_center().is_none());
        assert!(weights_only.historical_anchor_weights(0).is_none());
        assert!(weights_only.historical_anchor_weights(2).is_some());

        let _ = fs::remove_dir_all(&base_cont);
        let _ = fs::remove_dir_all(&base_res);
    }

    #[test]
    fn st04_round_id_overflow_fails_closed_without_mutating_state() {
        let base = temp_dir("st04_round_overflow");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x64; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let p1 = TestPeer::new(47);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::ModelDeltaV1 {
                server_learning_rate: 1.0,
            },
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1.peer_id].into_iter().collect(),
            min_quorum: 1,
            max_staleness: 0,
            max_clock_skew_secs: 60,
        };
        let mut coord = SwarmRoundCoordinator::initialize(
            run_id,
            config,
            linear_model_2_1(&[1.0, 2.0, 0.5]),
            &mut session,
            1_000,
        )
        .unwrap();
        coord.current_round = u64::MAX;

        let s1 = p1.establish_session(&mut coord, 1, 0x01, 100);
        let b = p1.build_dense_delivery(
            &coord,
            s1,
            1,
            1,
            100,
            u64::MAX,
            coord.parent_digest(),
            PayloadKindV1::ModelDeltaV1,
            &[0.1, 0.1, 0.1],
            0,
        );
        coord
            .submit_verified_delivery(&b, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100)
            .unwrap();
        let err = coord
            .finalize_round::<swarm_torch_models::optim::SgdOptimizer<LinearModel>, DummyEvaluator>(
                &[],
                None,
                None,
                &mut session,
                2_000,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            RoundError::RoundIdOverflow { round_id: u64::MAX }
        ));
        assert_eq!(coord.current_round(), u64::MAX);

        let _ = fs::remove_dir_all(&base);
    }
}
