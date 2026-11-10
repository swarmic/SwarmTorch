//! Content-Addressed Storage (CAS) checkpoint spine and recipe/content identity separation
//! (`ADR-0016`, `ADR-0023` §2.2, `P2-07`..`P2-09`, `W10-01`).
//!
//! Separates:
//! - [`CheckpointRecipeKeyV1`]: deterministic pre-execution lineage/recipe identity derived from
//!   `(run_id, round_id, node_def_hash, parent_recipe_key)`.
//! - [`CheckpointContentDigestV1`]: post-execution SHA-256 content digest over canonical
//!   serialized [`swarm_torch_models::ModelState`] bytes stored under
//!   `runs/<run_id>/blobs/sha256/<content_sha256_hex>`.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use swarm_torch_core::observe::RunId;
use swarm_torch_core::traits::SwarmModel;

use super::io::{
    decode_hex_exact, hex_lower, read_and_hash_bounded, read_json, write_json_pretty_atomic,
    write_once_cas_blob, MAX_SINGLE_ARTIFACT_BYTES,
};

/// Domain tag for deterministic checkpoint recipe keys (`ADR-0023` §2.2).
pub const CHECKPOINT_RECIPE_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.checkpoint.recipe.v1";

/// Schema version for [`RoundResumeStateV1`].
pub const ROUND_RESUME_STATE_SCHEMA_V1: u32 = 1;

/// Relative path inside `runs/<run_id>/` for durable round resume state.
pub const ROUND_RESUME_STATE_REL_PATH: &str = "checkpoints/round_resume_state.json";

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
