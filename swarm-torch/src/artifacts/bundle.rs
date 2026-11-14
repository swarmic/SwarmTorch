use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use swarm_torch_core::crypto::{
    KeyPair, MessageAuth, Signature, MANIFEST_ENTRIES_DOMAIN_TAG_V1,
    MANIFEST_SIGNATURE_DOMAIN_TAG_V1,
};
use swarm_torch_core::dataops::{
    DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1, LineageEdgeV1, MaterializationRecordV1,
    MaterializationRecordV2,
};
use swarm_torch_core::observe::{
    validate_event_record, validate_metric_record, validate_span_record, EventRecord, MetricRecord,
    RunId, SpanRecord,
};
use swarm_torch_core::run_graph::{validate_graph_v1, validate_node_v1, GraphV1};

use super::cas::{
    get_verified_coordinator_resume_state_v2, is_unhashed_checkpoint_head_rel_path,
    read_committed_head_v1, CheckpointContentDigestV1, CommittedHeadV1, CoordinatorResumeStateV2,
    COMMITTED_HEAD_REL_PATH,
};
use super::dp_ledger::DP_LEDGER_REL_PATH;
use super::io::{
    append_ndjson_with_policy, atomic_write, collect_files_recursive, decode_hex_exact,
    ensure_file, hex_lower, parse_json_slice, read_and_hash_bounded, read_bytes_bounded, read_json,
    rel_path_string, sha256_file, sync_parent_dir, write_json_pretty_atomic, DurabilityPolicyV1,
    MAX_MANIFEST_BYTES, MAX_SINGLE_ARTIFACT_BYTES, MAX_TOTAL_BUNDLE_BYTES,
};
use super::lock::{BundleLockGuard, LockAcquisitionPolicy};
use super::record_validation_error_to_io;

pub const SCHEMA_VERSION_V1: u32 = 1;

/// Domain-separated Ed25519 signature block for `manifest.json` (`ADR-0023` §3.1, `W10-02`).
///
/// Signs `"swarmtorch.manifest.sig.v1" || u32_le(schema_version) || run_id_raw_16_bytes || u64_le(signed_unix_nanos) || entries_hash`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestSignatureV1 {
    /// Signer's Ed25519 verifying key (64 lowercase hex chars = 32 bytes).
    pub signer_public_key: String,
    /// Signing timestamp in Unix nanoseconds.
    pub signed_unix_nanos: u64,
    /// Lowercase hex SHA-256 of canonical length-delimited `entries`.
    pub entries_hash: String,
    /// Lowercase hex Ed25519 signature (128 lowercase hex chars = 64 bytes).
    pub signature: String,
}

/// Top-level `manifest.json` contract (`ADR-0016`, `ADR-0023` §3.1).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestV1 {
    pub schema_version: u32,
    pub run_id: RunId,
    pub hash_algo: String,
    pub entries: Vec<ManifestEntryV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<ManifestSignatureV1>,
}

/// Single path-addressed entry in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestEntryV1 {
    /// Path relative to `runs/<run_id>/`.
    pub path: String,
    /// Lowercase hex SHA-256 digest (64 chars).
    pub sha256: String,
    /// File length in bytes.
    pub bytes: u64,
    /// Whether this path is required by the baseline v1 contract.
    pub required: bool,
}

/// Top-level `run.json` contract (`ADR-0016`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RunFileV1 {
    pub schema_version: u32,
    pub run_id: RunId,
    pub created_unix_nanos: u64,
    pub swarmtorch_version: String,
}

/// Consumer trust policy for artifact bundle loading, checkpoint restoration, and cache reuse
/// (`ADR-0023` §3.1, `P2-10`, `W10-02`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustPolicy {
    /// Verified-consumer default: requires an external trusted Ed25519 verifying key,
    /// valid [`ManifestSignatureV1`] over raw 16-byte [`RunId`], optional expected `RunId`
    /// and minimum `signed_unix_nanos` freshness anchor, and single-pass digest/size verification.
    Strict {
        /// Allowed Ed25519 verifying keys (`[u8; 32]`). Must be non-empty.
        trusted_public_keys: Vec<[u8; 32]>,
        /// Optional expected `RunId` pin.
        expected_run_id: Option<RunId>,
        /// Optional rollback protection floor on `signed_unix_nanos`.
        min_signed_unix_nanos: Option<u64>,
    },
    /// Explicit historical-inspection mode for unsigned bundles. Returns
    /// [`VerificationStatus::UnverifiedLegacyPermissive`] and cannot authorize verified
    /// checkpoint restore or strict cache admission.
    LegacyPermissive,
}

impl TrustPolicy {
    /// Construct a [`TrustPolicy::Strict`] policy with a single trusted public key.
    pub fn strict_with_key(public_key: [u8; 32]) -> Self {
        Self::Strict {
            trusted_public_keys: vec![public_key],
            expected_run_id: None,
            min_signed_unix_nanos: None,
        }
    }
}

/// Machine-readable verification status attached to a [`VerifiedBundleV1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VerificationStatus {
    /// Signed manifest verified against an authorized external key under [`TrustPolicy::Strict`].
    VerifiedStrict {
        /// Verified Ed25519 public key that signed the manifest.
        signer_public_key: [u8; 32],
        /// Signed timestamp in Unix nanoseconds.
        signed_unix_nanos: u64,
    },
    /// Loaded under [`TrustPolicy::LegacyPermissive`] without an external trust anchor.
    UnverifiedLegacyPermissive,
}

impl VerificationStatus {
    /// Returns `true` only when verified under [`TrustPolicy::Strict`].
    pub const fn is_strict(&self) -> bool {
        matches!(self, Self::VerifiedStrict { .. })
    }
}

/// Authenticated in-memory snapshot of a verified run artifact bundle (`ADR-0023` §3.1, `W10-02`).
///
/// Consumers read exclusively from [`VerifiedBundleV1::get_bytes`] / [`VerifiedBundleV1::require_bytes`],
/// eliminating TOCTOU path reopens and ignoring unlisted files on disk.
#[derive(Debug, Clone)]
pub struct VerifiedBundleV1 {
    run_dir: PathBuf,
    run_id: RunId,
    manifest: ManifestV1,
    status: VerificationStatus,
    files: BTreeMap<String, Vec<u8>>,
}

impl VerifiedBundleV1 {
    /// Root directory of the run bundle.
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Verified [`RunId`] of the bundle.
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Verified [`ManifestV1`].
    pub const fn manifest(&self) -> &ManifestV1 {
        &self.manifest
    }

    /// Verification status (`VerifiedStrict` vs. `UnverifiedLegacyPermissive`).
    pub const fn status(&self) -> VerificationStatus {
        self.status
    }

    /// Returns `true` if the bundle satisfied [`TrustPolicy::Strict`].
    pub const fn is_strict_verified(&self) -> bool {
        self.status.is_strict()
    }

    /// Look up verified bytes for a manifest-listed relative path (`/`-separated).
    ///
    /// Returns `None` if `rel_path` was not listed in the authenticated `manifest.json`
    /// (even if an unlisted file exists at that path on disk).
    pub fn get_bytes(&self, rel_path: &str) -> Option<&[u8]> {
        self.files.get(rel_path).map(Vec::as_slice)
    }

    /// Require verified bytes for a manifest-listed relative path.
    pub fn require_bytes(&self, rel_path: &str) -> io::Result<&[u8]> {
        self.get_bytes(rel_path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("file not present in verified manifest: {rel_path}"),
            )
        })
    }

    /// Iterate over all verified `(path, bytes)` entries.
    pub fn files(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.files
    }
}

/// Compute the canonical SHA-256 `entries_hash` over sorted [`ManifestEntryV1`] items.
pub fn compute_manifest_entries_hash_v1(entries: &[ManifestEntryV1]) -> io::Result<[u8; 32]> {
    let count = u32::try_from(entries.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many manifest entries"))?;
    let mut hasher = Sha256::new();
    hasher.update(MANIFEST_ENTRIES_DOMAIN_TAG_V1);
    hasher.update(count.to_le_bytes());

    for entry in entries {
        let path_bytes = entry.path.as_bytes();
        let path_len = u32::try_from(path_bytes.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "manifest entry path too long")
        })?;
        let sha_raw = decode_hex_exact::<32>(&entry.sha256)?;
        hasher.update(path_len.to_le_bytes());
        hasher.update(path_bytes);
        hasher.update(sha_raw);
        hasher.update(entry.bytes.to_le_bytes());
        hasher.update([u8::from(entry.required)]);
    }

    Ok(hasher.finalize().into())
}

/// Sign a [`ManifestV1`] in-place using `signer` (`ADR-0023` §3.1).
pub fn sign_manifest_v1(
    manifest: &mut ManifestV1,
    signer: &KeyPair,
    signed_unix_nanos: u64,
) -> io::Result<ManifestSignatureV1> {
    let entries_hash = compute_manifest_entries_hash_v1(&manifest.entries)?;
    let auth = MessageAuth::new(signer.clone());
    let sig = auth.sign_manifest_v1(
        manifest.schema_version,
        manifest.run_id.as_bytes(),
        signed_unix_nanos,
        &entries_hash,
    );
    let sig_block = ManifestSignatureV1 {
        signer_public_key: hex_lower(signer.public_key()),
        signed_unix_nanos,
        entries_hash: hex_lower(&entries_hash),
        signature: hex_lower(sig.as_bytes()),
    };
    manifest.signature = Some(sig_block.clone());
    Ok(sig_block)
}

/// Verify a [`ManifestSignatureV1`] on `manifest` against `entries_hash` (`ADR-0023` §3.1).
pub fn verify_manifest_signature_v1(
    manifest: &ManifestV1,
    entries_hash: &[u8; 32],
) -> io::Result<([u8; 32], u64)> {
    let _ = MANIFEST_SIGNATURE_DOMAIN_TAG_V1;
    let sig_block = manifest.signature.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "manifest signature is missing",
        )
    })?;
    let signer_pk = decode_hex_exact::<32>(&sig_block.signer_public_key)?;
    let claimed_entries_hash = decode_hex_exact::<32>(&sig_block.entries_hash)?;
    if &claimed_entries_hash != entries_hash {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "manifest signature entries_hash mismatch",
        ));
    }
    let sig_bytes = decode_hex_exact::<64>(&sig_block.signature)?;
    let sig = Signature::from_bytes(sig_bytes);
    MessageAuth::verify_manifest_v1(
        &signer_pk,
        manifest.schema_version,
        manifest.run_id.as_bytes(),
        sig_block.signed_unix_nanos,
        entries_hash,
        &sig,
    )
    .map_err(|e| {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("manifest signature verification failed: {e}"),
        )
    })?;
    Ok((signer_pk, sig_block.signed_unix_nanos))
}

/// A writer/validator for a single run artifact bundle (`runs/<run_id>/...`).
#[derive(Debug, Clone)]
pub struct RunArtifactBundle {
    run_dir: PathBuf,
    run_id: RunId,
}

impl RunArtifactBundle {
    /// Open an existing bundle directory (`runs/<run_id>/...`) by reading `run.json`.
    pub fn open(run_dir: impl AsRef<Path>) -> io::Result<Self> {
        let run_dir = run_dir.as_ref().to_path_buf();
        let run_file: RunFileV1 = read_json(&run_dir.join("run.json"))?;
        Ok(Self {
            run_dir,
            run_id: run_file.run_id,
        })
    }

    /// Create a new bundle directory at `<base>/runs/<run_id>/` with baseline v1 files.
    pub fn create(base: impl AsRef<Path>, run_id: RunId) -> io::Result<Self> {
        if !run_id.is_valid() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "run_id must be non-zero",
            ));
        }

        let run_id_hex = run_id.to_string();
        let run_dir = base.as_ref().join("runs").join(&run_id_hex);

        if run_dir.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "run artifact bundle already exists",
            ));
        }

        fs::create_dir_all(run_dir.join("datasets"))?;
        fs::create_dir_all(run_dir.join("artifacts"))?;
        fs::create_dir_all(run_dir.join("blobs").join("sha256"))?;

        let _lock = BundleLockGuard::acquire(&run_dir, LockAcquisitionPolicy::FailFast)?;

        // Baseline JSON files.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let created_unix_nanos = now.as_nanos().min(u64::MAX as u128) as u64;

        let run_file = RunFileV1 {
            schema_version: SCHEMA_VERSION_V1,
            run_id,
            created_unix_nanos,
            swarmtorch_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        write_json_pretty_atomic(&run_dir.join("run.json"), &run_file)?;

        let graph = GraphV1 {
            graph_id: Some(run_id.to_string()),
            ..GraphV1::default()
        };
        write_json_pretty_atomic(&run_dir.join("graph.json"), &graph)?;

        // DataOps baselines (ADR-0016).
        let registry = DatasetRegistryV1::default();
        write_json_pretty_atomic(&run_dir.join("datasets").join("registry.json"), &registry)?;

        let lineage = DatasetLineageV1::default();
        write_json_pretty_atomic(&run_dir.join("datasets").join("lineage.json"), &lineage)?;

        // NDJSON baselines (empty files are valid).
        ensure_file(&run_dir.join("spans.ndjson"))?;
        ensure_file(&run_dir.join("events.ndjson"))?;
        ensure_file(&run_dir.join("metrics.ndjson"))?;
        ensure_file(&run_dir.join("datasets").join("materializations.ndjson"))?;
        ensure_file(&run_dir.join("datasets").join("registry_updates.ndjson"))?;
        ensure_file(&run_dir.join("datasets").join("lineage_edges.ndjson"))?;

        let bundle = Self { run_dir, run_id };
        // Emit an initial manifest while holding the lock.
        bundle.finalize_manifest_unlocked(None)?;
        Ok(bundle)
    }

    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Acquire the cross-process exclusive `.bundle.lock` guard (`ADR-0023` §3.2, `P2-02`).
    pub fn acquire_lock(&self, policy: LockAcquisitionPolicy) -> io::Result<BundleLockGuard> {
        BundleLockGuard::acquire(&self.run_dir, policy).map_err(io::Error::from)
    }

    /// Write (replace) `graph.json` with a normalized graph.
    ///
    /// Validates node field bounds (M-09) and computes derived fields (`node_id`, `node_def_hash`).
    pub fn write_graph(&self, graph: &GraphV1) -> io::Result<()> {
        let _lock = self.acquire_lock(LockAcquisitionPolicy::default())?;
        self.write_graph_unlocked(graph)
    }

    pub(crate) fn write_graph_unlocked(&self, graph: &GraphV1) -> io::Result<()> {
        let mut g = graph.clone();
        for node in &g.nodes {
            validate_node_v1(node)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        }
        validate_graph_v1(&g)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        for node in &mut g.nodes {
            if node.code_ref.as_deref().unwrap_or("").is_empty() {
                node.code_ref = Some(format!("swarm-torch@{}", env!("CARGO_PKG_VERSION")));
            }
        }
        let normalized = g
            .normalize()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        validate_graph_v1(&normalized)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        write_json_pretty_atomic(&self.run_dir.join("graph.json"), &normalized)
    }

    pub fn append_span(&self, span: &SpanRecord) -> io::Result<()> {
        self.append_span_with_durability(span, DurabilityPolicyV1::SyncAtCommit)
    }

    pub fn append_span_with_durability(
        &self,
        span: &SpanRecord,
        durability: DurabilityPolicyV1,
    ) -> io::Result<()> {
        validate_span_record(span).map_err(record_validation_error_to_io)?;
        append_ndjson_with_policy(&self.run_dir.join("spans.ndjson"), span, durability)
    }

    pub fn append_event(&self, event: &EventRecord) -> io::Result<()> {
        self.append_event_with_durability(event, DurabilityPolicyV1::SyncAtCommit)
    }

    pub fn append_event_with_durability(
        &self,
        event: &EventRecord,
        durability: DurabilityPolicyV1,
    ) -> io::Result<()> {
        validate_event_record(event).map_err(record_validation_error_to_io)?;
        append_ndjson_with_policy(&self.run_dir.join("events.ndjson"), event, durability)
    }

    pub fn append_metric(&self, metric: &MetricRecord) -> io::Result<()> {
        self.append_metric_with_durability(metric, DurabilityPolicyV1::SyncAtCommit)
    }

    pub fn append_metric_with_durability(
        &self,
        metric: &MetricRecord,
        durability: DurabilityPolicyV1,
    ) -> io::Result<()> {
        validate_metric_record(metric).map_err(record_validation_error_to_io)?;
        append_ndjson_with_policy(&self.run_dir.join("metrics.ndjson"), metric, durability)
    }

    pub fn append_materialization(
        &self,
        materialization: &MaterializationRecordV1,
    ) -> io::Result<()> {
        append_ndjson_with_policy(
            &self
                .run_dir
                .join("datasets")
                .join("materializations.ndjson"),
            materialization,
            DurabilityPolicyV1::SyncEachAppend,
        )
    }

    pub fn append_materialization_v2(
        &self,
        materialization: &MaterializationRecordV2,
    ) -> io::Result<()> {
        append_ndjson_with_policy(
            &self
                .run_dir
                .join("datasets")
                .join("materializations.ndjson"),
            materialization,
            DurabilityPolicyV1::SyncEachAppend,
        )
    }

    pub fn append_registry_update(&self, dataset: &DatasetEntryV1) -> io::Result<()> {
        append_ndjson_with_policy(
            &self
                .run_dir
                .join("datasets")
                .join("registry_updates.ndjson"),
            dataset,
            DurabilityPolicyV1::SyncEachAppend,
        )
    }

    pub fn append_lineage_edge_update(&self, edge: &LineageEdgeV1) -> io::Result<()> {
        append_ndjson_with_policy(
            &self.run_dir.join("datasets").join("lineage_edges.ndjson"),
            edge,
            DurabilityPolicyV1::SyncEachAppend,
        )
    }

    pub fn write_dataset_registry(&self, registry: &DatasetRegistryV1) -> io::Result<()> {
        write_json_pretty_atomic(
            &self.run_dir.join("datasets").join("registry.json"),
            registry,
        )
    }

    pub fn write_dataset_lineage(&self, lineage: &DatasetLineageV1) -> io::Result<()> {
        write_json_pretty_atomic(&self.run_dir.join("datasets").join("lineage.json"), lineage)
    }

    /// Durability hook: flush and sync all required v1 files and `dp/ledger.ndjson` (if present),
    /// propagating any I/O failure (`ADR-0023` §2.2, `ST-05`).
    pub fn sync_required_v1(&self) -> io::Result<()> {
        for rel in required_paths_v1() {
            let path = self.run_dir.join(rel);
            let f = fs::OpenOptions::new().write(true).open(&path)?;
            f.sync_all()?;
        }
        let dp_ledger_path = self.run_dir.join(DP_LEDGER_REL_PATH);
        if dp_ledger_path.exists() {
            let f = fs::OpenOptions::new().write(true).open(&dp_ledger_path)?;
            f.sync_all()?;
            sync_parent_dir(&dp_ledger_path)?;
        }
        sync_parent_dir(&self.run_dir.join("run.json"))?;
        sync_parent_dir(&self.run_dir.join("datasets").join("registry.json"))?;
        Ok(())
    }

    /// (Re)compute and write an unsigned `manifest.json` for all current files in the bundle.
    pub fn finalize_manifest(&self) -> io::Result<()> {
        let _lock = self.acquire_lock(LockAcquisitionPolicy::default())?;
        self.finalize_manifest_unlocked(None).map(|_| ())
    }

    /// (Re)compute, sign with `signer`, and write `manifest.json` (`ADR-0023` §3.1, `W10-02`).
    pub fn finalize_signed_manifest(&self, signer: &KeyPair) -> io::Result<ManifestV1> {
        let _lock = self.acquire_lock(LockAcquisitionPolicy::default())?;
        self.finalize_manifest_unlocked(Some(signer))
    }

    /// Internal manifest writer for callers that already hold a [`BundleLockGuard`].
    pub(crate) fn finalize_manifest_unlocked(
        &self,
        signer: Option<&KeyPair>,
    ) -> io::Result<ManifestV1> {
        let (manifest, _, _) = self.finalize_manifest_with_sha256_unlocked(signer)?;
        Ok(manifest)
    }

    /// Internal manifest writer returning `(ManifestV1, raw_json_bytes, manifest_sha256_hex)`.
    pub(crate) fn finalize_manifest_with_sha256_unlocked(
        &self,
        signer: Option<&KeyPair>,
    ) -> io::Result<(ManifestV1, Vec<u8>, String)> {
        let canonical_root = self.run_dir.canonicalize()?;

        // Ensure baseline v1 required files exist and are synced before hashing.
        for p in required_paths_v1() {
            let full = self.run_dir.join(p);
            if !full.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing required bundle file: {p}"),
                ));
            }
        }
        self.sync_required_v1()?;

        let mut files = Vec::new();
        collect_files_recursive(&self.run_dir, &mut files)?;

        let mut entries = Vec::new();
        for file_path in files {
            if file_path.file_name().and_then(|s| s.to_str()) == Some("manifest.json") {
                continue;
            }
            let rel = rel_path_string(&file_path, &self.run_dir)?;
            if is_unhashed_checkpoint_head_rel_path(&rel) {
                continue;
            }
            validate_manifest_path(&rel)?;
            ensure_path_within_bundle(&file_path, &canonical_root, &rel)?;
            let bytes = fs::metadata(&file_path)?.len();
            let digest = sha256_file(&file_path)?;
            entries.push(ManifestEntryV1 {
                required: is_required_path_v1(&rel),
                path: rel,
                sha256: hex_lower(&digest),
                bytes,
            });
        }
        entries.sort_by(|a, b| a.path.cmp(&b.path));

        let mut manifest = ManifestV1 {
            schema_version: SCHEMA_VERSION_V1,
            run_id: self.run_id,
            hash_algo: "sha256".to_string(),
            entries,
            signature: None,
        };

        if let Some(key_pair) = signer {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();
            let signed_unix_nanos = now.as_nanos().min(u64::MAX as u128) as u64;
            sign_manifest_v1(&mut manifest, key_pair, signed_unix_nanos)?;
        }

        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(io::Error::other)?;
        let manifest_sha256 = hex_lower(&Sha256::digest(&manifest_bytes));
        atomic_write(&self.run_dir.join("manifest.json"), &manifest_bytes)?;
        Ok((manifest, manifest_bytes, manifest_sha256))
    }

    fn verify_manifest_header_and_policy(
        &self,
        manifest: &ManifestV1,
        policy: &TrustPolicy,
    ) -> io::Result<VerificationStatus> {
        if manifest.schema_version != SCHEMA_VERSION_V1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported manifest schema_version",
            ));
        }
        if manifest.run_id != self.run_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest run_id mismatch",
            ));
        }
        if manifest.hash_algo != "sha256" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported hash algorithm",
            ));
        }

        let entries_hash = compute_manifest_entries_hash_v1(&manifest.entries)?;

        match policy {
            TrustPolicy::Strict {
                trusted_public_keys,
                expected_run_id,
                min_signed_unix_nanos,
            } => {
                if trusted_public_keys.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "TrustPolicy::Strict requires at least one trusted public key",
                    ));
                }
                if let Some(expected_id) = expected_run_id {
                    if *expected_id != manifest.run_id {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            format!(
                                "manifest run_id {} does not match expected run_id {}",
                                manifest.run_id, expected_id
                            ),
                        ));
                    }
                }
                let (signer_pk, signed_unix_nanos) =
                    verify_manifest_signature_v1(manifest, &entries_hash)?;
                if !trusted_public_keys.iter().any(|k| k == &signer_pk) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "manifest signer public key is not in trusted_public_keys",
                    ));
                }
                if let Some(min_ts) = min_signed_unix_nanos {
                    if signed_unix_nanos < *min_ts {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            format!(
                                "manifest signed_unix_nanos {signed_unix_nanos} is older than minimum {min_ts}"
                            ),
                        ));
                    }
                }
                Ok(VerificationStatus::VerifiedStrict {
                    signer_public_key: signer_pk,
                    signed_unix_nanos,
                })
            }
            TrustPolicy::LegacyPermissive => {
                if manifest.signature.is_some() {
                    let _ = verify_manifest_signature_v1(manifest, &entries_hash)?;
                }
                Ok(VerificationStatus::UnverifiedLegacyPermissive)
            }
        }
    }

    /// Validate `manifest.json` and load all manifest-listed files in a single bounded pass
    /// (`ADR-0023` §3.1, `W10-02`).
    ///
    /// Eliminates TOCTOU path reopens and unlisted file consultation.
    pub fn verify_and_load(&self, policy: &TrustPolicy) -> io::Result<VerifiedBundleV1> {
        let manifest_path = self.run_dir.join("manifest.json");
        let manifest_bytes = read_bytes_bounded(&manifest_path, MAX_MANIFEST_BYTES)?;
        let manifest: ManifestV1 = parse_json_slice(&manifest_bytes)?;
        let canonical_root = self.run_dir.canonicalize()?;
        let status = self.verify_manifest_header_and_policy(&manifest, policy)?;

        let mut seen_paths: BTreeSet<String> = BTreeSet::new();
        let mut seen_required_paths: BTreeSet<String> = BTreeSet::new();
        let mut loaded_files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut total_bytes: u64 = 0;

        for entry in &manifest.entries {
            validate_manifest_path(&entry.path)?;

            if !seen_paths.insert(entry.path.clone()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("duplicate manifest path entry: {}", entry.path),
                ));
            }

            let required_by_schema = is_required_path_v1(&entry.path);
            if required_by_schema && !entry.required {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "required path marked non-required in manifest: {}",
                        entry.path
                    ),
                ));
            }
            if required_by_schema {
                seen_required_paths.insert(entry.path.clone());
            }

            total_bytes = total_bytes.checked_add(entry.bytes).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "total bundle byte size overflow",
                )
            })?;
            if total_bytes > MAX_TOTAL_BUNDLE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "total bundle bytes {total_bytes} exceed maximum {MAX_TOTAL_BUNDLE_BYTES}"
                    ),
                ));
            }

            let path = self.run_dir.join(&entry.path);
            if !path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing file listed in manifest: {}", entry.path),
                ));
            }
            ensure_path_within_bundle(&path, &canonical_root, &entry.path)?;

            let (bytes, digest) =
                read_and_hash_bounded(&path, entry.bytes, MAX_SINGLE_ARTIFACT_BYTES)?;
            let actual = hex_lower(&digest);
            if actual != entry.sha256 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("sha256 mismatch for {}", entry.path),
                ));
            }
            loaded_files.insert(entry.path.clone(), bytes);
        }

        for required in required_paths_v1() {
            if !seen_required_paths.contains(*required) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("required manifest entry missing: {required}"),
                ));
            }
        }

        // Verify that the loaded `run.json` agrees with `manifest.run_id`.
        if let Some(run_json_bytes) = loaded_files.get("run.json") {
            let run_file: RunFileV1 = parse_json_slice(run_json_bytes)?;
            if run_file.run_id != manifest.run_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "run.json run_id does not match manifest run_id",
                ));
            }
        }

        Ok(VerifiedBundleV1 {
            run_dir: self.run_dir.clone(),
            run_id: self.run_id,
            manifest,
            status,
            files: loaded_files,
        })
    }

    /// Validate `manifest.json` against current on-disk bytes.
    pub fn validate_manifest(&self) -> io::Result<()> {
        self.verify_and_load(&TrustPolicy::LegacyPermissive)
            .map(|_| ())
    }

    /// Authenticate `checkpoints/committed_head.json`, verify its pinned manifest (active
    /// `manifest.json` or crash-consistent archived generation manifest matching `head.manifest_sha256`),
    /// verify all listed CAS dependencies for [`CoordinatorResumeStateV2`], and return the
    /// verified resume envelope (`ST-04`).
    pub fn verify_committed_head(
        &self,
        policy: &TrustPolicy,
    ) -> io::Result<(
        CommittedHeadV1,
        ManifestV1,
        VerificationStatus,
        CoordinatorResumeStateV2,
    )> {
        let head = read_committed_head_v1(&self.run_dir)?;
        if head.run_id != self.run_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "committed_head.json run_id {} does not match bundle run_id {}",
                    head.run_id, self.run_id
                ),
            ));
        }

        let manifest_path = self.run_dir.join("manifest.json");
        let active_manifest_sha = sha256_file(&manifest_path).ok().map(|d| hex_lower(&d));
        let (manifest, status) = if active_manifest_sha.as_deref()
            == Some(head.manifest_sha256.as_str())
        {
            let verified = self.verify_and_load(policy)?;
            (verified.manifest, verified.status)
        } else {
            // Crash-recovery path: a subsequent round may have updated manifest.json before
            // committed_head.json advanced; verify the generation's archived manifest matching
            // the pinned `head.manifest_sha256` hash and policy signature instead.
            let archive_rel = super::cas::committed_manifest_archive_rel_path(head.generation);
            let archive_path = self.run_dir.join(&archive_rel);
            if !archive_path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "active manifest.json sha256 ({:?}) != committed_head.manifest_sha256 ({}) and archived manifest {} is missing",
                        active_manifest_sha, head.manifest_sha256, archive_rel
                    ),
                ));
            }
            let archive_bytes = read_bytes_bounded(&archive_path, MAX_MANIFEST_BYTES)?;
            let archive_sha = hex_lower(&Sha256::digest(&archive_bytes));
            if archive_sha != head.manifest_sha256 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "archived manifest {} sha256 {} != committed_head.manifest_sha256 {}",
                        archive_rel, archive_sha, head.manifest_sha256
                    ),
                ));
            }
            let archived: ManifestV1 = parse_json_slice(&archive_bytes)?;
            let status = self.verify_manifest_header_and_policy(&archived, policy)?;
            validate_manifest_entries_well_formed(&archived)?;
            (archived, status)
        };

        let (resume_v2, resume_digest) =
            get_verified_coordinator_resume_state_v2(&self.run_dir, &head.resume_state_cas_uri)?;
        if resume_v2.run_id != head.run_id
            || resume_v2.generation != head.generation
            || resume_v2.round_id != head.round_id
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "CoordinatorResumeStateV2 metadata disagrees with committed_head.json",
            ));
        }
        let expected_next_round = head.round_id.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "committed_head.json round_id overflows next_round_id",
            )
        })?;
        if resume_v2.next_round_id != expected_next_round {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "CoordinatorResumeStateV2 next_round_id {} != expected {}",
                    resume_v2.next_round_id, expected_next_round
                ),
            ));
        }

        // Every immutable CAS artifact referenced by the resume envelope MUST be authenticated in
        // the verified manifest (and physical blob SHA-256 verified before use).
        verify_cas_entry_in_manifest(&manifest, &resume_digest, None)?;
        verify_cas_entry_in_manifest(
            &manifest,
            &resume_v2.model_digest,
            Some(resume_v2.model_blob_bytes),
        )?;
        verify_cas_entry_in_manifest(
            &manifest,
            &resume_v2.historical_anchors_digest,
            Some(resume_v2.historical_anchors_bytes),
        )?;
        if let Some(opt_digest) = resume_v2.optimizer_state_digest.as_ref() {
            verify_cas_entry_in_manifest(&manifest, opt_digest, resume_v2.optimizer_state_bytes)?;
        }
        if let Some(cclip_digest) = resume_v2.cclip_center_digest.as_ref() {
            verify_cas_entry_in_manifest(&manifest, cclip_digest, resume_v2.cclip_center_bytes)?;
        }

        // Keep `COMMITTED_HEAD_REL_PATH` referenced cleanly for manifest consistency checks.
        let _ = COMMITTED_HEAD_REL_PATH;

        Ok((head, manifest, status, resume_v2))
    }
}

fn validate_manifest_entries_well_formed(manifest: &ManifestV1) -> io::Result<()> {
    let mut seen_paths = BTreeSet::new();
    let mut seen_required = BTreeSet::new();
    for entry in &manifest.entries {
        validate_manifest_path(&entry.path)?;
        if !seen_paths.insert(entry.path.clone()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate manifest path entry: {}", entry.path),
            ));
        }
        let req = is_required_path_v1(&entry.path);
        if req && !entry.required {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "required path marked non-required in manifest: {}",
                    entry.path
                ),
            ));
        }
        if req {
            seen_required.insert(entry.path.clone());
        }
    }
    for required in required_paths_v1() {
        if !seen_required.contains(*required) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("required manifest entry missing: {required}"),
            ));
        }
    }
    Ok(())
}

fn verify_cas_entry_in_manifest(
    manifest: &ManifestV1,
    digest: &CheckpointContentDigestV1,
    expected_bytes: Option<u64>,
) -> io::Result<()> {
    let rel_path = digest.rel_blob_path();
    let expected_hex = digest.to_hex();
    let entry = manifest
        .entries
        .iter()
        .find(|e| e.path == rel_path)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("CAS object {rel_path} is not authenticated by manifest"),
            )
        })?;
    if entry.sha256 != expected_hex {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "manifest entry digest {} does not match expected CAS digest {}",
                entry.sha256, expected_hex
            ),
        ));
    }
    if let Some(exp_len) = expected_bytes {
        if entry.bytes != exp_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "manifest entry byte length {} for {} does not match expected {}",
                    entry.bytes, rel_path, exp_len
                ),
            ));
        }
    }
    Ok(())
}

pub(crate) fn required_paths_v1() -> &'static [&'static str] {
    &[
        "run.json",
        "graph.json",
        "spans.ndjson",
        "events.ndjson",
        "metrics.ndjson",
        "datasets/registry.json",
        "datasets/lineage.json",
        "datasets/materializations.ndjson",
        "datasets/registry_updates.ndjson",
        "datasets/lineage_edges.ndjson",
    ]
}

fn is_required_path_v1(p: &str) -> bool {
    required_paths_v1().contains(&p)
}

pub(crate) fn validate_manifest_path(path: &str) -> io::Result<()> {
    if path.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "manifest path must be non-empty",
        ));
    }

    let parsed = Path::new(path);
    if parsed.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("manifest path must be relative: {path}"),
        ));
    }

    let mut has_normal_component = false;
    for component in parsed.components() {
        match component {
            Component::Normal(_) => {
                has_normal_component = true;
            }
            Component::ParentDir => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("manifest path must not contain '..': {path}"),
                ));
            }
            Component::CurDir => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("manifest path must not contain '.': {path}"),
                ));
            }
            Component::RootDir => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("manifest path must be relative: {path}"),
                ));
            }
            Component::Prefix(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("manifest path must be relative: {path}"),
                ));
            }
        }
    }

    if !has_normal_component {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("manifest path is invalid: {path}"),
        ));
    }

    Ok(())
}

fn ensure_path_within_bundle(path: &Path, canonical_root: &Path, rel: &str) -> io::Result<()> {
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(canonical_root) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("manifest path escapes bundle root: {rel}"),
        ));
    }
    Ok(())
}
