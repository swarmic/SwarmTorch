//! Content-Addressed Storage (CAS) checkpoint spine and recipe/content identity separation
//! (`ADR-0016`, `ADR-0023` §2.2, `P2-07`..`P2-09`, `W10-01`).
//!
//! Separates:
//! - [`CheckpointRecipeKeyV1`]: deterministic pre-execution lineage/recipe identity derived from
//!   `(run_id, round_id, node_def_hash, parent_recipe_key)`.
//! - [`CheckpointContentDigestV1`]: post-execution SHA-256 content digest over canonical
//!   serialized [`swarm_torch_models::ModelState`] bytes stored under
//!   `runs/<run_id>/blobs/sha256/<content_sha256_hex>`.

use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use swarm_torch_core::aggregation::{BucketingConfig, FedAsyncStalenessDescriptorV1};
use swarm_torch_core::observe::RunId;
use swarm_torch_core::traits::{OptimizerStateSnapshotV1, PeerId, SwarmModel};

use super::io::{
    atomic_write, decode_hex_exact, hex_lower, parse_json_slice, read_and_hash_bounded, read_json,
    write_json_pretty_atomic, write_once_cas_blob, MAX_SINGLE_ARTIFACT_BYTES,
};

/// Domain tag for deterministic checkpoint recipe keys (`ADR-0023` §2.2).
pub const CHECKPOINT_RECIPE_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.checkpoint.recipe.v1";

/// Schema version for [`RoundResumeStateV1`].
pub const ROUND_RESUME_STATE_SCHEMA_V1: u32 = 1;

/// Relative path inside `runs/<run_id>/` for durable round resume state.
pub const ROUND_RESUME_STATE_REL_PATH: &str = "checkpoints/round_resume_state.json";

/// Schema version for [`CommittedHeadV1`].
pub const COMMITTED_HEAD_SCHEMA_V1: u32 = 1;

/// Relative path inside `runs/<run_id>/` for the single authoritative committed checkpoint head.
pub const COMMITTED_HEAD_REL_PATH: &str = "checkpoints/committed_head.json";

/// Directory relative prefix storing archived per-generation manifests outside manifest entry hashing.
pub const CHECKPOINT_MANIFESTS_DIR_REL_PREFIX: &str = "checkpoints/manifests/";

/// Schema version for [`CenteredClippingCenterBlobV1`].
pub const CCLIP_CENTER_BLOB_SCHEMA_V1: u32 = 1;

/// Schema version for [`HistoricalAnchorsBlobV1`].
pub const HISTORICAL_ANCHORS_BLOB_SCHEMA_V1: u32 = 1;

/// Schema version for [`CoordinatorResumeStateV2`].
pub const COORDINATOR_RESUME_STATE_SCHEMA_V2: u32 = 2;

/// Deterministic recipe/lineage identity for a round checkpoint (`ADR-0023` §2.2).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CheckpointRecipeKeyV1(pub [u8; 32]);

impl CheckpointRecipeKeyV1 {
    /// Genesis (round 0) recipe key for a run before any round executes.
    pub fn genesis(run_id: RunId) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CHECKPOINT_RECIPE_DOMAIN_TAG_V1);
        hasher.update(run_id.as_bytes());
        hasher.update(0u64.to_le_bytes());
        hasher.update([0u8; 32]);
        hasher.update([0u8; 32]);
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..]);
        Self(out)
    }

    /// Derive the deterministic recipe key for `round_id`.
    pub fn derive(
        run_id: RunId,
        round_id: u64,
        node_def_hash: &[u8; 32],
        parent_recipe_key: &Self,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CHECKPOINT_RECIPE_DOMAIN_TAG_V1);
        hasher.update(run_id.as_bytes());
        hasher.update(round_id.to_le_bytes());
        hasher.update(node_def_hash);
        hasher.update(parent_recipe_key.0);
        let digest = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..]);
        Self(out)
    }

    /// Raw 32-byte digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase 64-character hex representation.
    pub fn to_hex(&self) -> String {
        hex_lower(&self.0)
    }

    /// Parse from a 64-character hex string.
    pub fn from_hex(s: &str) -> io::Result<Self> {
        Ok(Self(decode_hex_exact::<32>(s)?))
    }
}

impl fmt::Debug for CheckpointRecipeKeyV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CheckpointRecipeKeyV1({})", self.to_hex())
    }
}

impl fmt::Display for CheckpointRecipeKeyV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl serde::Serialize for CheckpointRecipeKeyV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> serde::Deserialize<'de> for CheckpointRecipeKeyV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Physical SHA-256 content digest of a serialized [`swarm_torch_models::ModelState`]
/// blob in `blobs/sha256/<hex>` (`ADR-0023` §2.2).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CheckpointContentDigestV1(pub [u8; 32]);

impl CheckpointContentDigestV1 {
    /// Compute the SHA-256 content digest of `bytes`.
    pub fn from_bytes_slice(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        let mut out = [0u8; 32];
        out.copy_from_slice(&digest[..]);
        Self(out)
    }

    /// Raw 32-byte SHA-256 digest.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase 64-character hex representation.
    pub fn to_hex(&self) -> String {
        hex_lower(&self.0)
    }

    /// Canonical CAS URI (`cas://sha256/<64_hex>`).
    pub fn to_cas_uri(&self) -> String {
        format!("cas://sha256/{}", self.to_hex())
    }

    /// Relative path inside `runs/<run_id>/` (`blobs/sha256/<64_hex>`).
    pub fn rel_blob_path(&self) -> String {
        format!("blobs/sha256/{}", self.to_hex())
    }

    /// Parse from a 64-character hex string.
    pub fn from_hex(s: &str) -> io::Result<Self> {
        Ok(Self(decode_hex_exact::<32>(s)?))
    }
}

impl fmt::Debug for CheckpointContentDigestV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CheckpointContentDigestV1({})", self.to_hex())
    }
}

impl fmt::Display for CheckpointContentDigestV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl serde::Serialize for CheckpointContentDigestV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> serde::Deserialize<'de> for CheckpointContentDigestV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Durable round resume metadata persisted at `checkpoints/round_resume_state.json`
/// (`ADR-0023` §2.2, `W10-01`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RoundResumeStateV1 {
    pub schema_version: u32,
    pub run_id: RunId,
    pub round_id: u64,
    pub checkpoint_asset_key: String,
    pub recipe_key: CheckpointRecipeKeyV1,
    pub content_digest: CheckpointContentDigestV1,
    pub blob_uri: String,
    pub blob_bytes: u64,
    pub parameter_count: usize,
    pub committed_unix_nanos: u64,
}

/// Explicit coordinator restore mode (`ADR-0023` §2.2, `ST-04`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RestoreModeV1 {
    /// Load only verified checkpoint model weights onto a clean training lineage
    /// (`cclip_center = None`, fresh replay onboarding, no optimizer state required).
    WeightsOnlyRestore,
    /// Full transactional state restore (`model`, `optimizer`, `cclip_center`, bounded
    /// `historical_anchors`, `coordinator_generation`, replay floor), failing closed if any
    /// required dependency or digest verification fails.
    ExactTrainingResume,
}

/// Authoritative atomic commit pointer persisted at `checkpoints/committed_head.json` (`ST-04`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommittedHeadV1 {
    pub schema_version: u32,
    pub run_id: RunId,
    pub generation: u64,
    pub round_id: u64,
    pub manifest_sha256: String,
    pub resume_state_cas_uri: String,
    pub committed_unix_nanos: u64,
}

/// Immutable CAS blob payload storing [`swarm_torch_core::aggregation::CenteredClipping`]
/// reference center $v^{(r-1)}$.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CenteredClippingCenterBlobV1 {
    pub schema_version: u32,
    pub center: Vec<f32>,
}

/// Immutable CAS blob payload storing bounded historical parameter anchors (`round_id -> weights`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HistoricalAnchorsBlobV1 {
    pub schema_version: u32,
    pub anchors: BTreeMap<u64, Vec<f32>>,
}

/// Complete durable coordinator snapshot persisted as an immutable CAS blob (`ST-04`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CoordinatorResumeStateV2 {
    pub schema_version: u32,
    pub run_id: RunId,
    pub swarm_instance_id: [u8; 16],
    pub coordinator_peer_id: PeerId,
    pub generation: u64,
    pub round_id: u64,
    pub next_round_id: u64,
    pub checkpoint_asset_key: String,
    pub recipe_key: CheckpointRecipeKeyV1,
    pub model_digest: CheckpointContentDigestV1,
    pub model_blob_uri: String,
    pub model_blob_bytes: u64,
    pub parameter_count: usize,
    pub optimizer_state_digest: Option<CheckpointContentDigestV1>,
    pub optimizer_state_cas_uri: Option<String>,
    pub optimizer_state_bytes: Option<u64>,
    pub cclip_center_digest: Option<CheckpointContentDigestV1>,
    pub cclip_center_cas_uri: Option<String>,
    pub cclip_center_bytes: Option<u64>,
    pub historical_anchors_digest: CheckpointContentDigestV1,
    pub historical_anchors_cas_uri: String,
    pub historical_anchors_bytes: u64,
    pub historical_digests: BTreeMap<u64, CheckpointContentDigestV1>,
    pub replay_timestamp_floor_secs: u32,
    pub config_policy_digest_hex: String,
    pub fedasync_policy: Option<FedAsyncStalenessDescriptorV1>,
    pub fedasync_hinge_threshold: u64,
    pub fedasync_clip_norm: Option<f32>,
    pub bucketing: Option<BucketingConfig>,
    pub committed_unix_nanos: u64,
}

/// Serialize `model` via [`swarm_torch_models::ModelState`] and publish it to
/// `runs/<run_id>/blobs/sha256/<hex_digest>` using write-once atomic semantics.
pub fn put_checkpoint_model<M: SwarmModel>(
    run_dir: &Path,
    model: &M,
) -> io::Result<(CheckpointContentDigestV1, String, u64)> {
    for &p in model.parameters() {
        if !p.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot persist checkpoint model containing non-finite parameters",
            ));
        }
    }
    let state = swarm_torch_models::ModelState::new(
        "swarmtorch.checkpoint.v1",
        model.parameters().to_vec(),
    );
    let bytes = state
        .to_bytes()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let digest = CheckpointContentDigestV1::from_bytes_slice(&bytes);
    write_once_cas_blob(run_dir, digest.as_bytes(), &bytes)?;
    let uri = digest.to_cas_uri();
    Ok((digest, uri, bytes.len() as u64))
}

/// Load and cryptographically verify a checkpoint model from `runs/<run_id>/blobs/sha256/<hex_digest>`
/// into `template`.
///
/// Fails closed if the CAS blob is missing, its SHA-256 digest does not match `expected_digest`,
/// its parameter count differs from `expected_param_count`, or any parameter is non-finite.
pub fn get_verified_checkpoint_model<M: SwarmModel>(
    run_dir: &Path,
    mut template: M,
    expected_digest: &CheckpointContentDigestV1,
    expected_param_count: usize,
) -> io::Result<M> {
    let blob_path = run_dir
        .join("blobs")
        .join("sha256")
        .join(expected_digest.to_hex());
    if !blob_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("checkpoint CAS blob not found: {}", blob_path.display()),
        ));
    }
    let meta_len = std::fs::metadata(&blob_path)?.len();
    let (bytes, actual_sha) =
        read_and_hash_bounded(&blob_path, meta_len, MAX_SINGLE_ARTIFACT_BYTES)?;
    if &actual_sha != expected_digest.as_bytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "checkpoint CAS blob SHA-256 mismatch for {}",
                expected_digest
            ),
        ));
    }
    let state = swarm_torch_models::ModelState::from_bytes(&bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    if state.version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported checkpoint ModelState version: {}",
                state.version
            ),
        ));
    }
    if state.parameters.len() != expected_param_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "checkpoint model parameter count {} != expected {}",
                state.parameters.len(),
                expected_param_count
            ),
        ));
    }
    if state.parameters.iter().any(|p: &f32| !p.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint model contains non-finite parameters",
        ));
    }
    template
        .load_parameters(&state.parameters)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(template)
}

/// Extract a 32-byte SHA-256 digest from a `"cas://sha256/<64_hex>"` or `"blobs/sha256/<64_hex>"` URI.
pub fn parse_cas_uri_digest(uri: &str) -> Option<[u8; 32]> {
    let hex_part = uri
        .strip_prefix("cas://sha256/")
        .or_else(|| uri.strip_prefix("blobs/sha256/"))?;
    decode_hex_exact::<32>(hex_part).ok()
}

/// Resolve and hash-verify a CAS URI against `runs/<run_id>/blobs/sha256/<64_hex>`.
pub fn verify_cas_uri_exists(
    run_dir: &Path,
    uri: &str,
    expected_bytes: Option<u64>,
) -> io::Result<PathBuf> {
    let expected_sha = parse_cas_uri_digest(uri).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid CAS URI: {uri}"),
        )
    })?;
    let hex_str = hex_lower(&expected_sha);
    let blob_path = run_dir.join("blobs").join("sha256").join(&hex_str);
    if !blob_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("CAS blob missing on disk: {}", blob_path.display()),
        ));
    }
    let len = match expected_bytes {
        Some(b) => b,
        None => std::fs::metadata(&blob_path)?.len(),
    };
    let (_, actual_sha) = read_and_hash_bounded(&blob_path, len, MAX_SINGLE_ARTIFACT_BYTES)?;
    if actual_sha != expected_sha {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("CAS blob {hex_str} failed SHA-256 content verification"),
        ));
    }
    Ok(blob_path)
}

/// Atomically persist [`RoundResumeStateV1`] under `checkpoints/round_resume_state.json`.
pub fn write_round_resume_state(run_dir: &Path, state: &RoundResumeStateV1) -> io::Result<()> {
    let path = run_dir.join(ROUND_RESUME_STATE_REL_PATH);
    write_json_pretty_atomic(&path, state)
}

/// Load [`RoundResumeStateV1`] from `checkpoints/round_resume_state.json` if present,
/// verifying that its referenced physical CAS blob exists and matches `state.content_digest`.
pub fn read_verified_round_resume_state(run_dir: &Path) -> io::Result<Option<RoundResumeStateV1>> {
    let path = run_dir.join(ROUND_RESUME_STATE_REL_PATH);
    if !path.exists() {
        return Ok(None);
    }
    let state: RoundResumeStateV1 = read_json(&path)?;
    if state.schema_version != ROUND_RESUME_STATE_SCHEMA_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported RoundResumeStateV1 schema_version: {}",
                state.schema_version
            ),
        ));
    }
    let expected_uri = state.content_digest.to_cas_uri();
    if state.blob_uri != expected_uri {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "RoundResumeStateV1 blob_uri {} does not match content_digest {}",
                state.blob_uri, expected_uri
            ),
        ));
    }
    verify_cas_uri_exists(run_dir, &state.blob_uri, Some(state.blob_bytes))?;
    Ok(Some(state))
}

/// Relative path inside `runs/<run_id>/` for an archived per-generation manifest snapshot.
pub(crate) fn committed_manifest_archive_rel_path(generation: u64) -> String {
    format!("{CHECKPOINT_MANIFESTS_DIR_REL_PREFIX}gen_{generation:020}.manifest.json")
}

/// Whether `rel` is a mutable checkpoint pointer or archived manifest excluded from manifest
/// entry hashing so publishing `checkpoints/committed_head.json` after Step 3 never causes circular
/// manifest invalidation (`ST-04`).
pub(crate) fn is_unhashed_checkpoint_head_rel_path(rel: &str) -> bool {
    rel == COMMITTED_HEAD_REL_PATH
        || rel == ROUND_RESUME_STATE_REL_PATH
        || rel.starts_with(CHECKPOINT_MANIFESTS_DIR_REL_PREFIX)
}

/// Persist canonical JSON bytes for `value` into write-once CAS storage (`blobs/sha256/<hex>`).
pub(crate) fn put_json_cas_blob<T: serde::Serialize>(
    run_dir: &Path,
    value: &T,
) -> io::Result<(CheckpointContentDigestV1, String, u64)> {
    let bytes = serde_json::to_vec(value)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    let digest = CheckpointContentDigestV1::from_bytes_slice(&bytes);
    write_once_cas_blob(run_dir, digest.as_bytes(), &bytes)?;
    let uri = digest.to_cas_uri();
    Ok((digest, uri, bytes.len() as u64))
}

/// Load and hash-verify a JSON CAS object stored at `blobs/sha256/<expected_digest>`.
pub(crate) fn get_verified_json_cas_blob<T: serde::de::DeserializeOwned>(
    run_dir: &Path,
    expected_digest: &CheckpointContentDigestV1,
    expected_bytes: Option<u64>,
    label: &str,
) -> io::Result<T> {
    let blob_path = run_dir
        .join("blobs")
        .join("sha256")
        .join(expected_digest.to_hex());
    if !blob_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{label} CAS blob not found: {}", blob_path.display()),
        ));
    }
    let meta_len = std::fs::metadata(&blob_path)?.len();
    if let Some(exp_bytes) = expected_bytes {
        if meta_len != exp_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{label} CAS blob byte size mismatch: expected {exp_bytes}, got {meta_len}"
                ),
            ));
        }
    }
    let (bytes, actual_sha) =
        read_and_hash_bounded(&blob_path, meta_len, MAX_SINGLE_ARTIFACT_BYTES)?;
    if &actual_sha != expected_digest.as_bytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} CAS blob SHA-256 mismatch for {expected_digest}"),
        ));
    }
    parse_json_slice(&bytes)
}

/// Persist [`OptimizerStateSnapshotV1`] into write-once CAS storage (`ST-04`).
pub fn put_optimizer_state_snapshot_v1(
    run_dir: &Path,
    snapshot: &OptimizerStateSnapshotV1,
) -> io::Result<(CheckpointContentDigestV1, String, u64)> {
    put_json_cas_blob(run_dir, snapshot)
}

/// Load and hash-verify [`OptimizerStateSnapshotV1`] from CAS (`ST-04`).
pub fn get_verified_optimizer_state_snapshot_v1(
    run_dir: &Path,
    expected_digest: &CheckpointContentDigestV1,
    expected_bytes: Option<u64>,
) -> io::Result<OptimizerStateSnapshotV1> {
    get_verified_json_cas_blob(run_dir, expected_digest, expected_bytes, "optimizer state")
}

/// Persist a [`swarm_torch_core::aggregation::CenteredClipping`] center vector into write-once CAS storage (`ST-04`).
pub fn put_cclip_center_v1(
    run_dir: &Path,
    center: &[f32],
) -> io::Result<(CheckpointContentDigestV1, String, u64)> {
    if center.iter().any(|v| !v.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot persist non-finite cclip_center vector",
        ));
    }
    let blob = CenteredClippingCenterBlobV1 {
        schema_version: CCLIP_CENTER_BLOB_SCHEMA_V1,
        center: center.to_vec(),
    };
    put_json_cas_blob(run_dir, &blob)
}

/// Load and validate a [`swarm_torch_core::aggregation::CenteredClipping`] center vector from CAS (`ST-04`).
pub fn get_verified_cclip_center_v1(
    run_dir: &Path,
    expected_digest: &CheckpointContentDigestV1,
    expected_bytes: Option<u64>,
    expected_param_count: usize,
) -> io::Result<Vec<f32>> {
    let blob: CenteredClippingCenterBlobV1 =
        get_verified_json_cas_blob(run_dir, expected_digest, expected_bytes, "cclip_center")?;
    if blob.schema_version != CCLIP_CENTER_BLOB_SCHEMA_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported cclip_center schema_version: {}",
                blob.schema_version
            ),
        ));
    }
    if blob.center.len() != expected_param_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "cclip_center dimension mismatch: expected {expected_param_count}, got {}",
                blob.center.len()
            ),
        ));
    }
    if blob.center.iter().any(|v| !v.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cclip_center CAS blob contains non-finite values",
        ));
    }
    Ok(blob.center)
}

/// Persist bounded historical parameter anchors (`round_id -> weights`) into CAS (`ST-04`).
pub fn put_historical_anchors_v1(
    run_dir: &Path,
    anchors: &BTreeMap<u64, Vec<f32>>,
) -> io::Result<(CheckpointContentDigestV1, String, u64)> {
    for weights in anchors.values() {
        if weights.iter().any(|v| !v.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot persist historical anchor containing non-finite values",
            ));
        }
    }
    let blob = HistoricalAnchorsBlobV1 {
        schema_version: HISTORICAL_ANCHORS_BLOB_SCHEMA_V1,
        anchors: anchors.clone(),
    };
    put_json_cas_blob(run_dir, &blob)
}

/// Load and validate bounded historical parameter anchors from CAS (`ST-04`).
pub fn get_verified_historical_anchors_v1(
    run_dir: &Path,
    expected_digest: &CheckpointContentDigestV1,
    expected_bytes: Option<u64>,
    expected_param_count: usize,
) -> io::Result<BTreeMap<u64, Vec<f32>>> {
    let blob: HistoricalAnchorsBlobV1 = get_verified_json_cas_blob(
        run_dir,
        expected_digest,
        expected_bytes,
        "historical_anchors",
    )?;
    if blob.schema_version != HISTORICAL_ANCHORS_BLOB_SCHEMA_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported historical_anchors schema_version: {}",
                blob.schema_version
            ),
        ));
    }
    for (round_id, weights) in &blob.anchors {
        if weights.len() != expected_param_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "historical anchor round {round_id} length {} != expected {expected_param_count}",
                    weights.len()
                ),
            ));
        }
        if weights.iter().any(|v| !v.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("historical anchor round {round_id} contains non-finite values"),
            ));
        }
    }
    Ok(blob.anchors)
}

/// Persist [`CoordinatorResumeStateV2`] into write-once CAS storage (`ST-04`).
pub fn put_coordinator_resume_state_v2(
    run_dir: &Path,
    state: &CoordinatorResumeStateV2,
) -> io::Result<(CheckpointContentDigestV1, String, u64)> {
    put_json_cas_blob(run_dir, state)
}

/// Load and schema/digest-validate [`CoordinatorResumeStateV2`] from CAS URI (`ST-04`).
pub fn get_verified_coordinator_resume_state_v2(
    run_dir: &Path,
    resume_state_cas_uri: &str,
) -> io::Result<(CoordinatorResumeStateV2, CheckpointContentDigestV1)> {
    let digest_bytes = parse_cas_uri_digest(resume_state_cas_uri).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid resume_state_cas_uri: {resume_state_cas_uri}"),
        )
    })?;
    let expected_digest = CheckpointContentDigestV1(digest_bytes);
    let state: CoordinatorResumeStateV2 =
        get_verified_json_cas_blob(run_dir, &expected_digest, None, "CoordinatorResumeStateV2")?;
    if state.schema_version != COORDINATOR_RESUME_STATE_SCHEMA_V2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported CoordinatorResumeStateV2 schema_version: {}",
                state.schema_version
            ),
        ));
    }
    Ok((state, expected_digest))
}

/// Atomically persist [`CommittedHeadV1`] under `checkpoints/committed_head.json` (`ST-04`).
pub fn write_committed_head_v1(run_dir: &Path, head: &CommittedHeadV1) -> io::Result<()> {
    let path = run_dir.join(COMMITTED_HEAD_REL_PATH);
    write_json_pretty_atomic(&path, head)
}

/// Archive raw bytes of a generation's finalized `manifest.json` outside manifest entry hashing
/// (`ST-04` crash recovery before `committed_head.json` update).
pub(crate) fn write_archived_generation_manifest_bytes(
    run_dir: &Path,
    generation: u64,
    manifest_bytes: &[u8],
) -> io::Result<()> {
    let rel = committed_manifest_archive_rel_path(generation);
    atomic_write(&run_dir.join(rel), manifest_bytes)
}

/// Read and structurally validate [`CommittedHeadV1`] from `checkpoints/committed_head.json`
/// (`ST-04`), returning `NotFound` if absent.
pub fn read_committed_head_v1(run_dir: &Path) -> io::Result<CommittedHeadV1> {
    let path = run_dir.join(COMMITTED_HEAD_REL_PATH);
    if !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("committed head pointer not found: {}", path.display()),
        ));
    }
    let head: CommittedHeadV1 = read_json(&path)?;
    if head.schema_version != COMMITTED_HEAD_SCHEMA_V1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported CommittedHeadV1 schema_version: {}",
                head.schema_version
            ),
        ));
    }
    decode_hex_exact::<32>(&head.manifest_sha256)?;
    if parse_cas_uri_digest(&head.resume_state_cas_uri).is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid resume_state_cas_uri in committed_head.json: {}",
                head.resume_state_cas_uri
            ),
        ));
    }
    Ok(head)
}

/// Export `model` as a `SwarmTorchCheckpointV1` SafeTensors v1 binary payload (`ADR-0024` §3, `W11-04`)
/// and publish it to `runs/<run_id>/blobs/sha256/<hex_digest>` using write-once atomic semantics.
pub fn put_safetensors_checkpoint_v1<M: SwarmModel>(
    run_dir: &Path,
    model: &M,
    schema: &swarm_torch_models::safetensors_v1::ModelTensorSchemaV1,
    limits: &swarm_torch_models::safetensors_v1::SafeTensorsLimitsV1,
) -> io::Result<(CheckpointContentDigestV1, String, u64)> {
    let bytes =
        swarm_torch_models::safetensors_v1::export_safetensors_checkpoint_v1(model, schema, limits)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    let digest = CheckpointContentDigestV1::from_bytes_slice(&bytes);
    write_once_cas_blob(run_dir, digest.as_bytes(), &bytes)?;
    let uri = digest.to_cas_uri();
    Ok((digest, uri, bytes.len() as u64))
}

/// Load and cryptographically verify a `SwarmTorchCheckpointV1` SafeTensors v1 checkpoint from
/// `runs/<run_id>/blobs/sha256/<hex_digest>` into `template` (`ADR-0024` §3, `W11-04`).
///
/// Fails closed if the CAS blob is missing, its SHA-256 digest does not match `expected_digest`,
/// its `model_schema_hash` or tensor layout differs from `expected_schema`, or any parameter is non-finite.
pub fn get_verified_safetensors_checkpoint_v1<M: SwarmModel>(
    run_dir: &Path,
    mut template: M,
    expected_digest: &CheckpointContentDigestV1,
    expected_schema: &swarm_torch_models::safetensors_v1::ModelTensorSchemaV1,
    limits: &swarm_torch_models::safetensors_v1::SafeTensorsLimitsV1,
) -> io::Result<M> {
    let blob_path = run_dir
        .join("blobs")
        .join("sha256")
        .join(expected_digest.to_hex());
    if !blob_path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "SafeTensors checkpoint CAS blob not found: {}",
                blob_path.display()
            ),
        ));
    }
    let meta_len = std::fs::metadata(&blob_path)?.len();
    let (bytes, actual_sha) =
        read_and_hash_bounded(&blob_path, meta_len, MAX_SINGLE_ARTIFACT_BYTES)?;
    if &actual_sha != expected_digest.as_bytes() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "SafeTensors checkpoint CAS blob SHA-256 mismatch for {}",
                expected_digest
            ),
        ));
    }
    swarm_torch_models::safetensors_v1::import_safetensors_checkpoint_v1(
        &mut template,
        expected_schema,
        &bytes,
        limits,
    )
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(template)
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_models::safetensors_v1::{ModelTensorSchemaV1, SafeTensorsLimitsV1};
    use swarm_torch_models::simple::LinearModel;

    #[test]
    fn safetensors_cas_roundtrip_and_serde_json_header_compatibility() {
        let temp_dir = std::env::temp_dir().join(format!(
            "swarmtorch_safetensors_cas_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp run dir");

        let mut model = LinearModel::new(3, 2).expect("valid linear model");
        model
            .load_parameters(&[0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.05, -0.05])
            .expect("load params");
        let schema = ModelTensorSchemaV1::for_linear(3, 2).expect("valid schema");
        let limits = SafeTensorsLimitsV1::default();

        let (digest, uri, size_bytes) =
            put_safetensors_checkpoint_v1(&temp_dir, &model, &schema, &limits)
                .expect("put_safetensors_checkpoint_v1 must succeed");
        assert!(uri.starts_with("cas://sha256/"));
        assert!(size_bytes > 8);

        // Verify the physical CAS file header is valid standard JSON via serde_json
        let blob_path = verify_cas_uri_exists(&temp_dir, &uri, Some(size_bytes))
            .expect("CAS blob must exist and match SHA-256");
        let raw_bytes = std::fs::read(&blob_path).expect("read CAS blob");
        let header_len = u64::from_le_bytes(raw_bytes[0..8].try_into().unwrap()) as usize;
        let header_json: serde_json::Value =
            serde_json::from_slice(&raw_bytes[8..8 + header_len]).expect("valid serde_json header");
        assert_eq!(
            header_json["__metadata__"]["format"].as_str(),
            Some("swarmtorch.checkpoint.v1")
        );
        assert!(header_json["__metadata__"]["model_schema_hash"]
            .as_str()
            .is_some());
        assert_eq!(header_json["linear.weight"]["dtype"].as_str(), Some("F32"));

        // Verify roundtrip into a fresh template model
        let template = LinearModel::new(3, 2).expect("template");
        let restored =
            get_verified_safetensors_checkpoint_v1(&temp_dir, template, &digest, &schema, &limits)
                .expect("get_verified_safetensors_checkpoint_v1 must succeed");
        assert_eq!(restored.parameters(), model.parameters());

        // Verify schema mismatch fails closed
        let wrong_schema =
            ModelTensorSchemaV1::single_f32_tensor("wrong.weights", 8).expect("wrong schema");
        let template2 = LinearModel::new(3, 2).expect("template2");
        assert!(get_verified_safetensors_checkpoint_v1(
            &temp_dir,
            template2,
            &digest,
            &wrong_schema,
            &limits,
        )
        .is_err());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
