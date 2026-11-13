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
    BucketExecutionReport, BucketingConfig, CenteredClipping, CoordinateMedian, FedAvg,
    PostBucketingRule, RobustAggregation, RobustAggregator, TrimmedMean,
};
#[cfg(feature = "robust-aggregation")]
use swarm_torch_core::aggregation::{Bulyan, Krum};
use swarm_torch_core::dataops::{
    AuthModeMarker, CacheDecisionV0, SourceDescriptorV0, TransformAuditV0, TrustClass,
};
use swarm_torch_core::observe::{
    AttrMap, AttrValue, EventRecord, MetricRecord, RunEventEmitter, RunId, SpanId, SpanRecord,
    TraceId,
};
use swarm_torch_core::replay::{CohortAllowlist, ReplayError, ReplayProtection};
use swarm_torch_core::run_graph::{
    node_def_hash_v1, node_id_from_key, AssetRefV1, CanonParams, ExecutionTrust, NodeV1, OpKind,
};
use swarm_torch_core::traits::{
    FitnessFunction, GradientUpdate, PeerId, SwarmModel, SwarmOptimizer, UpdateTransform,
};
use swarm_torch_core::Error as CoreError;

use crate::artifacts::{
    get_verified_checkpoint_model, pre_release_debit_dp_accountant, put_checkpoint_model,
    reconstruct_dp_accountant_from_ledger, write_round_resume_state, CheckpointContentDigestV1,
    CheckpointRecipeKeyV1, CryptoErrorOrIo, DataOpsSession, OutputSpec, RoundResumeStateV1,
    ROUND_RESUME_STATE_SCHEMA_V1,
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

/// Authenticated peer submission for a training round (`ADR-0023` §1.3).
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

/// Outcome of [`SwarmRoundCoordinator::submit_peer_update`] (`ADR-0023` §1.3).
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
            Self::Io(e) => write!(f, "artifact/CAS I/O error: {e}"),
        }
    }
}

impl std::error::Error for RoundError {}

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

/// Unified round coordinator state machine (`ADR-0023` §1, `W10-01`).
pub struct SwarmRoundCoordinator<M: SwarmModel> {
    run_id: RunId,
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
}

impl<M: SwarmModel + Clone> SwarmRoundCoordinator<M> {
    /// Initialize a new [`SwarmRoundCoordinator`] at round 1, persisting the genesis (round 0)
    /// model checkpoint into the run bundle's physical CAS store and registering it as a
    /// source asset in `session` (`ADR-0023` §1, §2.2).
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

        let allowlist = CohortAllowlist::from_peers(config.authorized_cohort.iter().copied());
        let replay = ReplayProtection::try_with_config(1024, config.max_clock_skew_secs)
            .expect("non-zero capacity")
            .with_cohort_allowlist(allowlist);

        let mut historical_digests = BTreeMap::new();
        historical_digests.insert(0, genesis_digest);
        let mut historical_anchors = BTreeMap::new();
        historical_anchors.insert(0, initial_model.parameters().to_vec());

        Ok(Self {
            run_id,
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
        })
    }

    /// Resume a [`SwarmRoundCoordinator`] from a verified [`RoundResumeStateV1`] and its
    /// physical CAS checkpoint blob (`ADR-0023` §2.2).
    pub fn from_resume_state(
        config: RoundCoordinatorConfig,
        model_template: M,
        resume_state: &RoundResumeStateV1,
        session: &DataOpsSession,
    ) -> Result<Self, RoundError> {
        config.validate()?;
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
        let allowlist = CohortAllowlist::from_peers(config.authorized_cohort.iter().copied());
        let replay = ReplayProtection::try_with_config(1024, config.max_clock_skew_secs)
            .expect("non-zero capacity")
            .with_cohort_allowlist(allowlist);

        let mut historical_digests = BTreeMap::new();
        historical_digests.insert(resume_state.round_id, resume_state.content_digest);
        let mut historical_anchors = BTreeMap::new();
        historical_anchors.insert(resume_state.round_id, model.parameters().to_vec());

        Ok(Self {
            run_id: resume_state.run_id,
            config,
            current_round: resume_state.round_id.saturating_add(1),
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
        })
    }

    /// Configure conservative `$s$-Bucketing` pre-aggregation for subsequent rounds (`ADR-0019` §4, `W11-01`).
    pub fn set_bucketing(&mut self, bucketing: Option<BucketingConfig>) {
        self.bucketing = bucketing;
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

    /// Validate and admit a peer update for the current round (`ADR-0023` §1.3).
    ///
    /// Enforces:
    /// 1. `run_id` binding.
    /// 2. Future-round check BEFORE unsigned subtraction (`submission.round_id > self.current_round`).
    /// 3. Bounded staleness check (`self.current_round - submission.round_id <= max_staleness`) and
    ///    `base_checkpoint_digest` binding.
    /// 4. `UpdateKindV1` compatibility check.
    /// 5. Cohort allowlist check BEFORE any state mutation.
    /// 6. Per-peer 1-vote-per-round equivocation vs. idempotent duplicate re-transmission check.
    /// 7. `ReplayProtection` sequence & timestamp validation.
    /// 8. Exact parameter dimension & finiteness validation.
    pub fn submit_peer_update(
        &mut self,
        submission: PeerRoundSubmissionV1,
        now_secs: u64,
    ) -> Result<SubmissionDecision, RoundError> {
        if submission.run_id != self.run_id {
            return Err(RoundError::RunIdMismatch {
                expected: self.run_id,
                got: submission.run_id,
            });
        }

        // Future-round check MUST precede any subtraction (`ADR-0023` §1.3).
        if submission.round_id > self.current_round {
            return Err(RoundError::FutureRoundRejected {
                submission_round: submission.round_id,
                current_round: self.current_round,
            });
        }
        let staleness = self.current_round - submission.round_id;
        if staleness > self.config.max_staleness {
            return Err(RoundError::StaleRoundRejected {
                submission_round: submission.round_id,
                current_round: self.current_round,
                max_staleness: self.config.max_staleness,
            });
        }

        let expected_base = if staleness == 0 {
            self.parent_digest
        } else {
            let prev_round = submission.round_id.saturating_sub(1);
            self.historical_digests
                .get(&prev_round)
                .copied()
                .unwrap_or(self.parent_digest)
        };
        if submission.base_checkpoint_digest != expected_base {
            return Err(RoundError::BaseCheckpointMismatch {
                expected: expected_base,
                got: submission.base_checkpoint_digest,
            });
        }

        if submission.update_kind != self.config.update_kind {
            return Err(RoundError::UpdateKindMismatch {
                expected: self.config.update_kind,
                got: submission.update_kind,
            });
        }

        if !self.config.authorized_cohort.contains(&submission.sender) {
            return Err(RoundError::UnauthorizedPeer(submission.sender));
        }

        if self.equivocated.contains(&submission.sender) {
            return Err(RoundError::EquivocationRejected {
                sender: submission.sender,
                round_id: self.current_round,
            });
        }

        let payload_digest = submission.canonical_payload_digest();
        if let Some(existing) = self.admitted.get(&submission.sender) {
            if existing.payload_digest == payload_digest {
                return Ok(SubmissionDecision::DuplicateIdempotentAck);
            } else {
                // Conflicting payload from the same peer in the same round: revoke their earlier vote!
                self.admitted.remove(&submission.sender);
                self.equivocated.insert(submission.sender);
                return Err(RoundError::EquivocationRejected {
                    sender: submission.sender,
                    round_id: self.current_round,
                });
            }
        }

        let expected_dim = self.model.num_parameters();
        if submission.values.len() != expected_dim {
            return Err(RoundError::DimensionMismatch {
                expected: expected_dim,
                got: submission.values.len(),
            });
        }
        for (idx, &val) in submission.values.iter().enumerate() {
            if !val.is_finite() {
                return Err(RoundError::NonFiniteInput {
                    sender: submission.sender,
                    coord_index: idx,
                });
            }
        }

        let ts_u32 = u32::try_from(submission.timestamp_secs).map_err(|_| {
            RoundError::ReplayRejected(ReplayError::Expired {
                ts: u32::MAX,
                now: u32::try_from(now_secs).unwrap_or(u32::MAX),
                window: self.config.max_clock_skew_secs,
            })
        })?;
        let now_u32 = u32::try_from(now_secs).map_err(|_| {
            RoundError::ReplayRejected(ReplayError::Expired {
                ts: ts_u32,
                now: u32::MAX,
                window: self.config.max_clock_skew_secs,
            })
        })?;
        self.replay
            .validate(&submission.sender, submission.sequence, ts_u32, now_u32)
            .map_err(RoundError::ReplayRejected)?;

        let grad_update = GradientUpdate {
            sender: *submission.sender.as_bytes(),
            sequence: submission.sequence,
            gradients: submission.values,
            round_id: self.current_round,
        };
        self.admitted.insert(
            submission.sender,
            AdmittedPeerEntry {
                payload_digest,
                update: grad_update,
                reported_examples: submission.reported_examples,
            },
        );

        Ok(SubmissionDecision::Admitted {
            admitted_count: self.admitted.len(),
            min_quorum: self.config.min_quorum,
        })
    }

    /// Finalize the current round, applying `transforms`, running the configured
    /// [`RobustAggregation`] / [`WeightingPolicyV1`], updating the global model, evaluating
    /// optional `fitness`, and committing the new checkpoint to the physical CAS store and
    /// [`DataOpsSession`] (`ADR-0023` §1, §2.2).
    pub fn finalize_round<O, F>(
        &mut self,
        transforms: &[&dyn UpdateTransform],
        mut server_optimizer: Option<&mut O>,
        fitness: Option<&F>,
        session: &mut DataOpsSession,
        ts_unix_nanos: u64,
    ) -> Result<RoundCommitSummaryV1, RoundError>
    where
        O: SwarmOptimizer<Model = M>,
        F: FitnessFunction<Model = M>,
    {
        // Re-verify quorum after any equivocation removals or timeout expiration (`ADR-0023` §1.3).
        if self.admitted.len() < self.config.min_quorum {
            return Err(RoundError::InsufficientQuorum {
                admitted: self.admitted.len(),
                min_quorum: self.config.min_quorum,
            });
        }

        // Deterministic PeerId-sorted order (`BTreeMap` iteration is sorted by `PeerId`).
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

        // Apply pre-aggregation transforms on cloned working buffers while preserving frozen_raw_inputs.
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
            session.record_transform_applied(&audit);
            working_updates = next_updates;
        }

        // Aggregate updates according to `WeightingPolicyV1`, optional `$s$-Bucketing`, and `RobustAggregation`.
        let (aggregated_vec, bucketing_report) =
            self.aggregate_working_updates(&admitted_peers, &working_updates)?;

        // Produce candidate model according to `UpdateKindV1`.
        let mut candidate_model = self.model.clone();
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
                let opt = server_optimizer.as_mut().ok_or_else(|| {
                    RoundError::OptimizerStep(
                        "UpdateKindV1::GradientV1 requires a server SwarmOptimizer".to_string(),
                    )
                })?;
                opt.step(&mut candidate_model, &aggregated_vec)
                    .map_err(|e| RoundError::OptimizerStep(format!("{e:?}")))?;
                if candidate_model.parameters().iter().any(|p| !p.is_finite()) {
                    return Err(RoundError::Aggregation(CoreError::NanDetected));
                }
            }
        }

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

        // Authoritative CAS checkpoint persistence + DataOps materialization (`ADR-0023` §2.2).
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        self.pre_release_dp_debit_if_enabled(
            session,
            self.current_round,
            ts_unix_nanos,
            None,
            None,
        )?;
        let (content_digest, blob_uri, blob_bytes) =
            put_checkpoint_model(&run_dir, &candidate_model)?;

        let checkpoint_asset_key = format!("checkpoint://round/{}", self.current_round);
        let node_key = format!("train/round_{}", self.current_round);
        let round_node = NodeV1 {
            node_key: node_key.clone(),
            node_id: Some(node_id_from_key(&node_key)),
            op_kind: OpKind::Train,
            op_type: "swarm_round_aggregate".to_string(),
            inputs: vec![AssetRefV1 {
                asset_key: self.parent_asset_key.clone(),
                fingerprint: Some(self.parent_digest.to_hex()),
            }],
            outputs: vec![AssetRefV1 {
                asset_key: checkpoint_asset_key.clone(),
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
            self.current_round,
            &node_def_digest,
            &self.parent_recipe_key,
        );

        session.materialize_node_outputs(
            &round_node,
            &[OutputSpec {
                asset_key: checkpoint_asset_key.clone(),
                schema: None,
                rows: Some(admitted_peers.len() as u64),
                bytes: Some(blob_bytes),
            }],
            ts_unix_nanos,
            CacheDecisionV0::Miss,
            1,
        )?;
        session.bind_output_cas_uri(
            &checkpoint_asset_key,
            &blob_uri,
            "application/x-swarmtorch-model-v1",
        )?;

        let resume_state = RoundResumeStateV1 {
            schema_version: ROUND_RESUME_STATE_SCHEMA_V1,
            run_id: self.run_id,
            round_id: self.current_round,
            checkpoint_asset_key: checkpoint_asset_key.clone(),
            recipe_key,
            content_digest,
            blob_uri: blob_uri.clone(),
            blob_bytes,
            parameter_count: candidate_model.num_parameters(),
            committed_unix_nanos: ts_unix_nanos,
        };
        write_round_resume_state(&run_dir, &resume_state)?;

        // Emit telemetry span, event, and optional fitness metric.
        let trace_id = TraceId::from_bytes(*self.run_id.as_bytes());
        let mut span_hasher = Sha256::new();
        span_hasher.update(b"swarmtorch.round.span.v1");
        span_hasher.update(self.run_id.as_bytes());
        span_hasher.update(self.current_round.to_le_bytes());
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
            AttrValue::I64(self.current_round as i64),
        );
        attrs.insert(
            "swarmtorch.admitted_peers".to_string(),
            AttrValue::I64(admitted_peers.len() as i64),
        );
        attrs.insert(
            "swarmtorch.equivocated_peers".to_string(),
            AttrValue::I64(self.equivocated.len() as i64),
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
            name: format!("round/{}", self.current_round),
            start_unix_nanos: ts_unix_nanos.saturating_sub(1_000_000),
            end_unix_nanos: Some(ts_unix_nanos),
            attrs: attrs.clone(),
        })?;
        session.sink().emit_event(&EventRecord {
            schema_version: 1,
            ts_unix_nanos,
            trace_id,
            span_id: Some(span_id),
            name: "coordinator/round_committed".to_string(),
            attrs: attrs.clone(),
        })?;
        if let Some(score) = fitness_score {
            session.sink().emit_metric(&MetricRecord {
                schema_version: 1,
                ts_unix_nanos,
                trace_id,
                span_id: Some(span_id),
                name: "round_fitness".to_string(),
                value: score as f64,
                unit: Some("score".to_string()),
                attrs,
            })?;
        }

        let equivocated_peers: Vec<PeerId> = self.equivocated.iter().copied().collect();
        let committed_round = self.current_round;

        // Advance coordinator state to the next round.
        self.model = candidate_model;
        self.parent_digest = content_digest;
        self.parent_recipe_key = recipe_key;
        self.parent_asset_key = checkpoint_asset_key.clone();
        self.historical_digests
            .insert(committed_round, content_digest);
        self.cclip_center = Some(aggregated_vec);
        self.last_bucketing_report = bucketing_report;
        self.record_anchor_and_prune(self.current_round, self.model.parameters().to_vec());
        self.current_round = self.current_round.saturating_add(1);
        self.admitted.clear();
        self.equivocated.clear();

        Ok(RoundCommitSummaryV1 {
            round_id: committed_round,
            admitted_peers,
            equivocated_peers,
            checkpoint_asset_key,
            recipe_key,
            content_digest,
            blob_uri,
            fitness_score,
            frozen_raw_inputs,
            weight_mismatch_warnings,
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

    fn record_anchor_and_prune(&mut self, round_id: u64, weights: Vec<f32>) {
        self.historical_anchors.insert(round_id, weights);
        let max_keep = self.config.max_staleness.max(
            self.fedasync_policy
                .map(|p| p.max_staleness_rounds)
                .unwrap_or(0),
        );
        let cutoff = round_id.saturating_sub(max_keep);
        self.historical_anchors.retain(|&r, _| r >= cutoff);
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

    /// Configure a durable Rényi Differential Privacy accountant backed by `dp/ledger.ndjson` (`ADR-0023` §4.2, `W12-03`).
    ///
    /// Replays any already-committed entries from `run_dir/dp/ledger.ndjson` so a crash or restart
    /// never refunds previously debited privacy budget.
    pub fn configure_dp_accountant(
        &mut self,
        run_dir: &std::path::Path,
        profile: swarm_torch_core::dp::DpMechanismProfileV1,
    ) -> Result<(), RoundError> {
        let acc =
            reconstruct_dp_accountant_from_ledger(run_dir, profile).map_err(|err| match err {
                CryptoErrorOrIo::Crypto(c) => RoundError::PrivacyBudget(c),
                CryptoErrorOrIo::Io(e) => RoundError::Io(e),
            })?;
        self.dp_accountant = Some(acc);
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
        let latest = self.current_round.saturating_sub(1);
        if !self.historical_anchors.contains_key(&latest) {
            self.historical_anchors
                .insert(latest, self.model.parameters().to_vec());
        }
        self.fedasync_policy = Some(policy);
        self.fedasync_hinge_threshold = hinge_threshold_rounds;
        self.fedasync_clip_norm = clip_norm;
        Ok(())
    }

    /// Lookup retained server parameter vector for historical anchor round `round_id`, if still in window.
    pub fn historical_anchor_weights(&self, round_id: u64) -> Option<&[f32]> {
        self.historical_anchors.get(&round_id).map(|v| v.as_slice())
    }

    /// Execute one bounded-staleness asynchronous (`FedAsync`) submission step against historical anchor
    /// round `submission.round_id`, applying pre-drift $\ell_2$ clipping, exact server drift compensation,
    /// and mandatory pre-release DP ledger debit BEFORE CAS checkpoint publication (`ADR-0023` §4.1-4.2, `W12-03`).
    pub fn step_fedasync_stale_submission(
        &mut self,
        submission: PeerRoundSubmissionV1,
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
        let policy = self
            .fedasync_policy
            .ok_or(RoundError::Aggregation(CoreError::InvalidThreshold))?;
        if submission.run_id != self.run_id {
            return Err(RoundError::RunIdMismatch {
                expected: self.run_id,
                got: submission.run_id,
            });
        }
        if !matches!(submission.update_kind, UpdateKindV1::ModelDeltaV1 { .. }) {
            return Err(RoundError::UpdateKindMismatch {
                expected: self.config.update_kind,
                got: submission.update_kind,
            });
        }

        let latest_committed = self.current_round.saturating_sub(1);
        if submission.round_id > latest_committed {
            return Err(RoundError::FutureRoundRejected {
                submission_round: submission.round_id,
                current_round: latest_committed,
            });
        }
        let staleness = latest_committed - submission.round_id;
        let max_stale = policy.max_staleness_rounds.min(self.config.max_staleness);
        if staleness > max_stale {
            return Err(RoundError::StaleRoundRejected {
                submission_round: submission.round_id,
                current_round: latest_committed,
                max_staleness: max_stale,
            });
        }

        let expected_base = self
            .historical_digests
            .get(&submission.round_id)
            .copied()
            .ok_or(RoundError::StaleAnchorEvicted {
                base_round: submission.round_id,
                current_round: latest_committed,
            })?;
        if submission.base_checkpoint_digest != expected_base {
            return Err(RoundError::BaseCheckpointMismatch {
                expected: expected_base,
                got: submission.base_checkpoint_digest,
            });
        }
        let w_base = self
            .historical_anchors
            .get(&submission.round_id)
            .ok_or(RoundError::StaleAnchorEvicted {
                base_round: submission.round_id,
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

        let ts_u32 = u32::try_from(submission.timestamp_secs).map_err(|_| {
            RoundError::ReplayRejected(ReplayError::Expired {
                ts: u32::MAX,
                now: u32::try_from(now_secs).unwrap_or(u32::MAX),
                window: self.config.max_clock_skew_secs,
            })
        })?;
        let now_u32 = u32::try_from(now_secs).unwrap_or(u32::MAX);
        self.replay
            .validate(&submission.sender, submission.sequence, ts_u32, now_u32)
            .map_err(RoundError::ReplayRejected)?;

        let outcome = swarm_torch_core::aggregation::apply_fedasync_stale_delta_v1(
            &HingePolicy {
                inner: policy,
                hinge: self.fedasync_hinge_threshold,
            },
            submission.round_id,
            latest_committed,
            self.model.parameters(),
            &w_base,
            &submission.values,
            self.fedasync_clip_norm,
        )
        .map_err(RoundError::Aggregation)?;

        let mut candidate_model = self.model.clone();
        candidate_model
            .load_parameters(&outcome.updated_weights)
            .map_err(RoundError::Aggregation)?;

        let committed_round = self.current_round;
        // Mandatory fail-closed DP pre-release debit BEFORE writing checkpoint to CAS!
        self.pre_release_dp_debit_if_enabled(
            session,
            committed_round,
            ts_unix_nanos,
            self.fedasync_clip_norm,
            Some(subsampling_rate_q),
        )?;

        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let (content_digest, blob_uri, blob_bytes) =
            put_checkpoint_model(&run_dir, &candidate_model)?;

        let checkpoint_asset_key = format!("checkpoint://round/{committed_round}");
        let node_key = format!("train/round_{committed_round}_fedasync");
        let round_node = NodeV1 {
            node_key: node_key.clone(),
            node_id: Some(node_id_from_key(&node_key)),
            op_kind: OpKind::Train,
            op_type: "fedasync_stale_step".to_string(),
            inputs: vec![AssetRefV1 {
                asset_key: self.parent_asset_key.clone(),
                fingerprint: Some(self.parent_digest.to_hex()),
            }],
            outputs: vec![AssetRefV1 {
                asset_key: checkpoint_asset_key.clone(),
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
            committed_round,
            &node_def_digest,
            &self.parent_recipe_key,
        );

        session.materialize_node_outputs(
            &round_node,
            &[OutputSpec {
                asset_key: checkpoint_asset_key.clone(),
                schema: None,
                rows: Some(1),
                bytes: Some(blob_bytes),
            }],
            ts_unix_nanos,
            CacheDecisionV0::Miss,
            1,
        )?;
        session.bind_output_cas_uri(
            &checkpoint_asset_key,
            &blob_uri,
            "application/x-swarmtorch-model-v1",
        )?;

        let resume_state = RoundResumeStateV1 {
            schema_version: ROUND_RESUME_STATE_SCHEMA_V1,
            run_id: self.run_id,
            round_id: committed_round,
            checkpoint_asset_key: checkpoint_asset_key.clone(),
            recipe_key,
            content_digest,
            blob_uri: blob_uri.clone(),
            blob_bytes,
            parameter_count: candidate_model.num_parameters(),
            committed_unix_nanos: ts_unix_nanos,
        };
        write_round_resume_state(&run_dir, &resume_state)?;

        let mut frozen_raw_inputs = BTreeMap::new();
        frozen_raw_inputs.insert(submission.sender, submission.values);

        self.model = candidate_model;
        self.parent_digest = content_digest;
        self.parent_recipe_key = recipe_key;
        self.parent_asset_key = checkpoint_asset_key.clone();
        self.historical_digests
            .insert(committed_round, content_digest);
        self.record_anchor_and_prune(committed_round, self.model.parameters().to_vec());
        self.current_round = committed_round.saturating_add(1);

        let summary = RoundCommitSummaryV1 {
            round_id: committed_round,
            admitted_peers: vec![submission.sender],
            equivocated_peers: Vec::new(),
            checkpoint_asset_key,
            recipe_key,
            content_digest,
            blob_uri,
            fitness_score: None,
            frozen_raw_inputs,
            weight_mismatch_warnings: Vec::new(),
        };
        Ok((outcome, summary))
    }

    /// Coordinate a cohort of LoRA / heterogeneous-rank FLoRA factor updates under authorized cohort
    /// gating, quorum checks, optional exact base-weight merge, and pre-release DP ledger debit (`ADR-0024` §4, `W12-01`).
    #[allow(clippy::too_many_arguments)]
    pub fn aggregate_lora_cohort_round(
        &mut self,
        submissions: &[(PeerId, swarm_torch_models::lora::LoraAdapterPairV1)],
        mode: swarm_torch_models::lora::LoraAggregationModeV1,
        b_factor_aggregator: &dyn swarm_torch_core::aggregation::RobustAggregator,
        base_weights_to_merge: Option<&mut [f32]>,
        subsampling_rate_q: f64,
        session: &mut DataOpsSession,
        ts_unix_nanos: u64,
    ) -> Result<swarm_torch_models::lora::LoraAggregationOutcomeV1, RoundError> {
        let mut seen_peers = BTreeSet::new();
        for (peer, _) in submissions {
            if !self.config.authorized_cohort.contains(peer) {
                return Err(RoundError::ReplayRejected(ReplayError::UnauthorizedPeer {
                    peer: *peer,
                }));
            }
            if !seen_peers.insert(*peer) {
                return Err(RoundError::Aggregation(CoreError::InvalidGradient));
            }
        }
        if submissions.len() < self.config.min_quorum {
            return Err(RoundError::InsufficientQuorum {
                admitted: submissions.len(),
                min_quorum: self.config.min_quorum,
            });
        }

        let outcome = swarm_torch_models::lora::aggregate_lora_adapters_v1(
            submissions,
            mode,
            b_factor_aggregator,
        )
        .map_err(|_| RoundError::Aggregation(CoreError::InvalidGradient))?;

        if let Some(base_w) = base_weights_to_merge {
            if let swarm_torch_models::lora::LoraAggregationOutcomeV1::FloraStacked { stacked } =
                &outcome
            {
                stacked
                    .apply_exact_to_base_weights(base_w)
                    .map_err(|_| RoundError::Aggregation(CoreError::InvalidGradient))?;
            }
        }

        // Fail-closed DP debit before releasing aggregated adapter factors.
        self.pre_release_dp_debit_if_enabled(
            session,
            self.current_round,
            ts_unix_nanos,
            None,
            Some(subsampling_rate_q),
        )?;

        Ok(outcome)
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

        let p1 = peer(1);
        let p2 = peer(2);
        let cohort: BTreeSet<PeerId> = [p1, p2].into_iter().collect();
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

        // Submit future round u64::MAX -> must return FutureRoundRejected without underflow panic!
        let future_sub = PeerRoundSubmissionV1 {
            run_id,
            round_id: u64::MAX,
            base_checkpoint_digest: coord.parent_digest(),
            sender: p1,
            sequence: 1,
            timestamp_secs: 100,
            update_kind: UpdateKindV1::default(),
            values: vec![0.1, 0.1, 0.1],
            reported_examples: None,
        };
        let err = coord.submit_peer_update(future_sub, 100).unwrap_err();
        assert!(matches!(
            err,
            RoundError::FutureRoundRejected {
                submission_round: u64::MAX,
                current_round: 1
            }
        ));

        // Submit stale round 0 when max_staleness == 0 -> StaleRoundRejected!
        let stale_sub = PeerRoundSubmissionV1 {
            run_id,
            round_id: 0,
            base_checkpoint_digest: coord.parent_digest(),
            sender: p1,
            sequence: 1,
            timestamp_secs: 100,
            update_kind: UpdateKindV1::default(),
            values: vec![0.1, 0.1, 0.1],
            reported_examples: None,
        };
        let err = coord.submit_peer_update(stale_sub, 100).unwrap_err();
        assert!(matches!(err, RoundError::StaleRoundRejected { .. }));

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

        let p1 = peer(1);
        let p2 = peer(2);
        let cohort: BTreeSet<PeerId> = [p1, p2].into_iter().collect();
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

        let sub1 = PeerRoundSubmissionV1 {
            run_id,
            round_id: 1,
            base_checkpoint_digest: coord.parent_digest(),
            sender: p1,
            sequence: 1,
            timestamp_secs: 100,
            update_kind: UpdateKindV1::default(),
            values: vec![0.2, 0.4, 0.6],
            reported_examples: None,
        };
        assert_eq!(
            coord.submit_peer_update(sub1.clone(), 100).unwrap(),
            SubmissionDecision::Admitted {
                admitted_count: 1,
                min_quorum: 2
            }
        );

        // Byte-identical re-transmission from p1 -> DuplicateIdempotentAck (still 1 vote!).
        assert_eq!(
            coord.submit_peer_update(sub1, 100).unwrap(),
            SubmissionDecision::DuplicateIdempotentAck
        );
        assert_eq!(coord.admitted_count(), 1);

        // Admit p2 so quorum (2) is reached.
        let sub2 = PeerRoundSubmissionV1 {
            run_id,
            round_id: 1,
            base_checkpoint_digest: coord.parent_digest(),
            sender: p2,
            sequence: 1,
            timestamp_secs: 100,
            update_kind: UpdateKindV1::default(),
            values: vec![0.4, 0.2, 0.0],
            reported_examples: None,
        };
        coord.submit_peer_update(sub2, 100).unwrap();
        assert!(coord.has_quorum());

        // Now p1 equivocates by sending a different vector in round 1!
        let conflicting_sub1 = PeerRoundSubmissionV1 {
            run_id,
            round_id: 1,
            base_checkpoint_digest: coord.parent_digest(),
            sender: p1,
            sequence: 2,
            timestamp_secs: 100,
            update_kind: UpdateKindV1::default(),
            values: vec![-9.0, -9.0, -9.0],
            reported_examples: None,
        };
        let err = coord.submit_peer_update(conflicting_sub1, 100).unwrap_err();
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

        let p1 = peer(10);
        let p2 = peer(20);
        let cohort: BTreeSet<PeerId> = [p1, p2].into_iter().collect();

        // Reject RegistryCertified paired with TrimmedMean
        let mut bad_weights = BTreeMap::new();
        bad_weights.insert(p1, 100);
        bad_weights.insert(p2, 300);
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

        // p1 sends raw delta [2.0, 2.0, 2.0] with spoofed reported_examples = 999_999
        coord
            .submit_peer_update(
                PeerRoundSubmissionV1 {
                    run_id,
                    round_id: 1,
                    base_checkpoint_digest: coord.parent_digest(),
                    sender: p1,
                    sequence: 1,
                    timestamp_secs: 100,
                    update_kind: config.update_kind,
                    values: vec![2.0, 2.0, 2.0],
                    reported_examples: Some(999_999),
                },
                100,
            )
            .unwrap();

        // p2 sends raw delta [0.2, 0.2, 0.2]
        coord
            .submit_peer_update(
                PeerRoundSubmissionV1 {
                    run_id,
                    round_id: 1,
                    base_checkpoint_digest: coord.parent_digest(),
                    sender: p2,
                    sequence: 1,
                    timestamp_secs: 100,
                    update_kind: config.update_kind,
                    values: vec![0.2, 0.2, 0.2],
                    reported_examples: Some(300),
                },
                100,
            )
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
        assert_eq!(summary.frozen_raw_inputs[&p1], vec![2.0, 2.0, 2.0]);
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

        let resumed_coord: SwarmRoundCoordinator<LinearModel> =
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

        let peers: Vec<PeerId> = (1..=6u8).map(peer).collect();
        let cohort: BTreeSet<PeerId> = peers.iter().copied().collect();
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

        // Round 1: 5 honest peers submit [0.2, 0.2, 0.2], 1 Byzantine peer submits [100.0, 100.0, 100.0].
        for (i, &p) in peers.iter().enumerate() {
            let vals = if i < 5 {
                vec![0.2, 0.2, 0.2]
            } else {
                vec![100.0, 100.0, 100.0]
            };
            coord
                .submit_peer_update(
                    PeerRoundSubmissionV1 {
                        run_id,
                        round_id: 1,
                        base_checkpoint_digest: coord.parent_digest(),
                        sender: p,
                        sequence: 1,
                        timestamp_secs: 100,
                        update_kind: UpdateKindV1::default(),
                        values: vals,
                        reported_examples: None,
                    },
                    100,
                )
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
        for (i, &p) in peers.iter().enumerate() {
            let vals = if i < 5 {
                vec![0.22, 0.22, 0.22]
            } else {
                vec![-500.0, -500.0, -500.0]
            };
            coord
                .submit_peer_update(
                    PeerRoundSubmissionV1 {
                        run_id,
                        round_id: 2,
                        base_checkpoint_digest: coord.parent_digest(),
                        sender: p,
                        sequence: 2,
                        timestamp_secs: 110,
                        update_kind: UpdateKindV1::default(),
                        values: vals,
                        reported_examples: None,
                    },
                    110,
                )
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

        let p1 = peer(1);
        let p2 = peer(2);
        let cohort: BTreeSet<PeerId> = [p1, p2].into_iter().collect();
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

        // Advance synchronous round 1: move weights to [2.0, 3.0, 1.5]
        for p in [p1, p2] {
            coord
                .submit_peer_update(
                    PeerRoundSubmissionV1 {
                        run_id,
                        round_id: 1,
                        base_checkpoint_digest: digest_r0,
                        sender: p,
                        sequence: 1,
                        timestamp_secs: 100,
                        update_kind: UpdateKindV1::ModelDeltaV1 {
                            server_learning_rate: 1.0,
                        },
                        values: vec![1.0, 1.0, 1.0],
                        reported_examples: None,
                    },
                    100,
                )
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
        let (outcome, _sum2) = coord
            .step_fedasync_stale_submission(
                PeerRoundSubmissionV1 {
                    run_id,
                    round_id: 0,
                    base_checkpoint_digest: digest_r0,
                    sender: p2,
                    sequence: 2,
                    timestamp_secs: 105,
                    update_kind: UpdateKindV1::ModelDeltaV1 {
                        server_learning_rate: 1.0,
                    },
                    values: vec![1000.0, 0.0, 0.0],
                    reported_examples: None,
                },
                105,
                0.5,
                &mut session,
                3_000,
            )
            .unwrap();
        assert!(outcome.was_clipped);
        assert!((outcome.clipped_delta_l2_norm - 1.0).abs() < 1e-4);
        assert_eq!(outcome.staleness_rounds, 1);

        // After advancing one more step, round 0 anchor was evicted (latest_committed = 2, cutoff = 1)
        assert!(coord.historical_anchor_weights(0).is_none());
        let stale_err = coord
            .step_fedasync_stale_submission(
                PeerRoundSubmissionV1 {
                    run_id,
                    round_id: 0,
                    base_checkpoint_digest: digest_r0,
                    sender: p1,
                    sequence: 3,
                    timestamp_secs: 106,
                    update_kind: UpdateKindV1::ModelDeltaV1 {
                        server_learning_rate: 1.0,
                    },
                    values: vec![0.1, 0.1, 0.1],
                    reported_examples: None,
                },
                106,
                0.5,
                &mut session,
                4_000,
            )
            .unwrap_err();
        assert!(matches!(stale_err, RoundError::StaleRoundRejected { .. }));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn dp_pre_release_ledger_debit_is_non_refundable_and_idempotent_across_crash_recovery() {
        use swarm_torch_core::crypto::CryptoError;
        use swarm_torch_core::dp::{
            DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1,
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

        let p1 = peer(1);
        let p2 = peer(2);
        let p_intruder = peer(9);
        let config = RoundCoordinatorConfig {
            update_kind: UpdateKindV1::default(),
            aggregation: RobustAggregation::FedAvg,
            weighting_policy: WeightingPolicyV1::EqualPeerVote,
            authorized_cohort: [p1, p2].into_iter().collect(),
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
                },
            )
            .unwrap();

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

        // Intruder peer rejected before DP debit
        assert!(matches!(
            coord.aggregate_lora_cohort_round(
                &[(p1, pair1.clone()), (p_intruder, pair2.clone())],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 16
                },
                &FedAvg,
                None,
                1.0,
                &mut session,
                2_000,
            ),
            Err(RoundError::ReplayRejected(
                ReplayError::UnauthorizedPeer { .. }
            ))
        ));
        assert_eq!(coord.dp_accountant().unwrap().committed_rounds(), 0);

        // Authorized heterogeneous-rank FLoRA aggregation merges exact base weight and debits DP ledger
        let mut base_w = vec![0.0f32; 4];
        let outcome = coord
            .aggregate_lora_cohort_round(
                &[(p1, pair1), (p2, pair2)],
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
}
