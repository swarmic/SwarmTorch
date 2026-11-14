use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use swarm_torch_core::crypto::KeyPair;
use swarm_torch_core::dataops::{
    cache_hit_from_decision, cache_key_v0, dataset_fingerprint_v0, derived_source_fingerprint_v0,
    no_schema_hash_v0, predict_output_fingerprints, recipe_hash_v0, sanitize_source_descriptor_v0,
    schema_hash_v0, source_fingerprint_v0, AuthModeMarker, CacheDecisionV0, DatasetEntryV1,
    DatasetLineageV1, DatasetRegistryV1, LineageEdgeV1, MaterializationRecordCompat,
    MaterializationRecordV2, MaterializationStatusV0, OutputSpecCore, PredictedOutput,
    SchemaDescriptorV0, SourceDescriptorV0, TransformAuditV0, TrustClass, UnsafeReasonV0,
    DATAOPS_SCHEMA_V1, MATERIALIZATION_SCHEMA_V2,
};
use swarm_torch_core::execution::AssetInstanceV1;
use swarm_torch_core::observe::RunId;
use swarm_torch_core::run_graph::{
    node_def_hash_v1, node_id_from_key, CachePolicyV0, ExecutionTrust, NodeV1,
};

use super::bundle::ManifestV1;
use super::cas::{
    parse_cas_uri_digest, read_verified_round_resume_state, verify_cas_uri_exists,
    RoundResumeStateV1,
};
use super::io::{
    hex_lower, parse_json_slice, parse_ndjson_slice_bounded, read_bytes_bounded, sha256_file,
    write_json_pretty_atomic, MAX_NDJSON_LINES, MAX_NDJSON_LINE_BYTES, MAX_SINGLE_ARTIFACT_BYTES,
};
use super::lock::{BundleLockGuard, LockAcquisitionPolicy};
use super::{ArtifactWriteProfile, RunArtifactBundle, RunArtifactSink, SnapshotProfile};

/// Error type for `DataOpsSession::predict()`.
///
/// Returned when prediction cannot proceed due to missing/invalid inputs.
/// Both variants include the asset_key for auditability.
#[derive(Debug)]
pub enum PredictError {
    /// An input asset_key declared by the node is not in the registry.
    MissingInput(String),
    /// An input's fingerprint in the registry is not valid hex.
    InvalidFingerprint(String),
    /// Prediction output list violated node output contract.
    OutputContract(String),
}

/// Output specification for `materialize_node_outputs`.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub asset_key: String,
    pub schema: Option<SchemaDescriptorV0>,
    pub rows: Option<u64>,
    pub bytes: Option<u64>,
}

/// Receipt returned when a [`DataOpsSession`] is durably committed and sealed (`ADR-0023` §2.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionCommitReceiptV1 {
    pub run_id: RunId,
    pub sealed_generation: u64,
    pub manifest: ManifestV1,
}

const SNAPSHOT_PAIR_SCHEMA_V1: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SnapshotPairCommitV1 {
    schema_version: u32,
    pair_seq: u64,
    registry_sha256: String,
    lineage_sha256: String,
}

/// DataOps session: manages registry/lineage with trust propagation, cross-process lock
/// coordination, physical CAS verification, and crash-safe persistence (`ADR-0023` §2).
#[derive(Debug)]
pub struct DataOpsSession {
    sink: Arc<RunArtifactSink>,
    /// asset_key -> DatasetEntryV1 (uniqueness enforced)
    registry: BTreeMap<String, DatasetEntryV1>,
    /// (input_fp, output_fp, node_id_str) -> LineageEdgeV1 (dedupe key)
    lineage: BTreeMap<(String, String, String), LineageEdgeV1>,
    /// Monotonic materialization record sequence number.
    next_record_seq: u64,
    /// Snapshot persistence profile for registry/lineage snapshots.
    snapshot_profile: SnapshotProfile,
    /// Monotonic snapshot pair commit marker sequence.
    next_snapshot_pair_seq: u64,
    /// Monotonic DataOps mutation counter used for streaming compaction cadence.
    dataops_write_count: u64,
    /// Transform audits to attach to the next materialization record(s).
    pending_transform_audits: Vec<TransformAuditV0>,
    /// Latest recorded physical `(rows, bytes)` keyed by `(asset_key, fingerprint_v0)`.
    cached_output_metrics: BTreeMap<(String, String), (Option<u64>, Option<u64>)>,
}

impl DataOpsSession {
    /// Create a new session wrapping an artifact sink.
    pub fn new(sink: Arc<RunArtifactSink>) -> Self {
        let profile = sink.profile().snapshot_profile;
        Self::with_profile(sink, profile)
    }

    /// Create a new session with an explicit snapshot profile.
    pub fn with_profile(sink: Arc<RunArtifactSink>, snapshot_profile: SnapshotProfile) -> Self {
        Self {
            sink,
            registry: BTreeMap::new(),
            lineage: BTreeMap::new(),
            next_record_seq: 1,
            snapshot_profile,
            next_snapshot_pair_seq: 1,
            dataops_write_count: 0,
            pending_transform_audits: Vec::new(),
            cached_output_metrics: BTreeMap::new(),
        }
    }

    /// Create a new run bundle on disk, acquire an exclusive `.bundle.lock`, and return
    /// an active [`DataOpsSession`] (`ADR-0023` §2.1, `W10-01`).
    pub fn create(
        base_dir: impl AsRef<Path>,
        run_id: RunId,
        profile: ArtifactWriteProfile,
        lock_policy: LockAcquisitionPolicy,
    ) -> io::Result<Self> {
        let bundle = RunArtifactBundle::create(base_dir, run_id)?;
        let lock_guard = bundle.acquire_lock(lock_policy)?;
        let snapshot_profile = profile.snapshot_profile;
        let sink = Arc::new(RunArtifactSink::with_profile_and_lock(
            bundle, profile, lock_guard,
        ));
        Ok(Self::with_profile(sink, snapshot_profile))
    }

    /// Open an existing run bundle directory, acquire an exclusive `.bundle.lock`, and
    /// replay persisted registry, lineage, and materialization state (`ADR-0023` §2.1, `W10-01`).
    pub fn open(
        run_dir: impl AsRef<Path>,
        profile: ArtifactWriteProfile,
        lock_policy: LockAcquisitionPolicy,
    ) -> io::Result<Self> {
        let run_dir = run_dir.as_ref();
        let lock_guard = BundleLockGuard::acquire(run_dir, lock_policy).map_err(io::Error::from)?;
        let bundle = RunArtifactBundle::open(run_dir)?;
        let snapshot_profile = profile.snapshot_profile;

        let datasets_dir = run_dir.join("datasets");
        let reg_snap_bytes = read_bytes_bounded(
            &datasets_dir.join("registry.json"),
            MAX_SINGLE_ARTIFACT_BYTES,
        )?;
        let reg_snap: DatasetRegistryV1 = parse_json_slice(&reg_snap_bytes)?;

        let reg_updates_path = datasets_dir.join("registry_updates.ndjson");
        let reg_updates: Vec<DatasetEntryV1> = if reg_updates_path.exists() {
            let bytes = read_bytes_bounded(&reg_updates_path, MAX_SINGLE_ARTIFACT_BYTES)?;
            parse_ndjson_slice_bounded(&bytes, MAX_NDJSON_LINE_BYTES, MAX_NDJSON_LINES)?
        } else {
            Vec::new()
        };

        let mut registry = BTreeMap::new();
        for entry in reg_snap.datasets {
            registry.insert(entry.asset_key.clone(), entry);
        }
        for entry in reg_updates {
            registry.insert(entry.asset_key.clone(), entry);
        }

        let lin_snap_bytes = read_bytes_bounded(
            &datasets_dir.join("lineage.json"),
            MAX_SINGLE_ARTIFACT_BYTES,
        )?;
        let lin_snap: DatasetLineageV1 = parse_json_slice(&lin_snap_bytes)?;

        let lin_updates_path = datasets_dir.join("lineage_edges.ndjson");
        let lin_updates: Vec<LineageEdgeV1> = if lin_updates_path.exists() {
            let bytes = read_bytes_bounded(&lin_updates_path, MAX_SINGLE_ARTIFACT_BYTES)?;
            parse_ndjson_slice_bounded(&bytes, MAX_NDJSON_LINE_BYTES, MAX_NDJSON_LINES)?
        } else {
            Vec::new()
        };

        let mut lineage = BTreeMap::new();
        for edge in lin_snap.edges.into_iter().chain(lin_updates) {
            let key = (
                edge.input_fingerprint_v0.clone(),
                edge.output_fingerprint_v0.clone(),
                edge.node_id.to_string(),
            );
            lineage.insert(key, edge);
        }

        let mat_path = datasets_dir.join("materializations.ndjson");
        let mut max_seq = 0u64;
        let mut cached_output_metrics = BTreeMap::new();
        if mat_path.exists() {
            let bytes = read_bytes_bounded(&mat_path, MAX_SINGLE_ARTIFACT_BYTES)?;
            let rows: Vec<MaterializationRecordCompat> =
                parse_ndjson_slice_bounded(&bytes, MAX_NDJSON_LINE_BYTES, MAX_NDJSON_LINES)?;
            for (idx, row) in rows.into_iter().enumerate() {
                let v2 = row.into_v2();
                if v2.status == MaterializationStatusV0::Ok
                    && (v2.rows.is_some() || v2.bytes.is_some())
                {
                    cached_output_metrics.insert(
                        (v2.asset_key.clone(), v2.fingerprint_v0.clone()),
                        (v2.rows, v2.bytes),
                    );
                }
                let seq = if v2.record_seq == 0 {
                    (idx as u64) + 1
                } else {
                    v2.record_seq
                };
                max_seq = max_seq.max(seq);
            }
        }

        let pair_marker_path = datasets_dir.join("snapshot_pair_commit.json");
        let next_snapshot_pair_seq = if pair_marker_path.exists() {
            let bytes = read_bytes_bounded(&pair_marker_path, MAX_SINGLE_ARTIFACT_BYTES)?;
            let marker: SnapshotPairCommitV1 = parse_json_slice(&bytes)?;
            marker.pair_seq.saturating_add(1)
        } else {
            1
        };

        let sink = Arc::new(RunArtifactSink::with_profile_and_lock(
            bundle, profile, lock_guard,
        ));
        Ok(Self {
            sink,
            registry,
            lineage,
            next_record_seq: max_seq.saturating_add(1),
            snapshot_profile,
            next_snapshot_pair_seq,
            dataops_write_count: 0,
            pending_transform_audits: Vec::new(),
            cached_output_metrics,
        })
    }

    /// Open an existing run bundle and load its verified [`RoundResumeStateV1`] if present
    /// (`ADR-0023` §2.2, `W10-01`).
    pub fn resume(
        run_dir: impl AsRef<Path>,
        profile: ArtifactWriteProfile,
        lock_policy: LockAcquisitionPolicy,
    ) -> io::Result<(Self, Option<RoundResumeStateV1>)> {
        let run_dir_ref = run_dir.as_ref();
        let session = Self::open(run_dir_ref, profile, lock_policy)?;
        let resume_state = read_verified_round_resume_state(run_dir_ref)?;
        Ok((session, resume_state))
    }

    /// Flush snapshots, sync required v1 files, finalize (and optionally sign) `manifest.json`,
    /// seal the underlying [`RunArtifactSink`] against late writes from cloned `Arc` handles,
    /// and release `.bundle.lock` (`ADR-0023` §2.1, `W10-01`).
    pub fn commit_and_close(
        mut self,
        signer: Option<&KeyPair>,
    ) -> io::Result<SessionCommitReceiptV1> {
        self.flush_snapshots()?;
        let (sealed_generation, manifest) = self.sink.seal_and_finalize(signer)?;
        Ok(SessionCommitReceiptV1 {
            run_id: self.sink.bundle().run_id(),
            sealed_generation,
            manifest,
        })
    }

    /// Record an applied update transform for the next materialization emission.
    ///
    /// These audits are attached to the next `materialize_node_outputs` call and then cleared.
    pub fn record_transform_applied(&mut self, audit: &TransformAuditV0) {
        self.pending_transform_audits.push(audit.clone());
    }

    pub(crate) fn pending_transform_audit_len(&self) -> usize {
        self.pending_transform_audits.len()
    }

    pub(crate) fn truncate_pending_transform_audits(&mut self, len: usize) {
        self.pending_transform_audits.truncate(len);
    }

    /// Look up fingerprint (64-char hex) for an asset_key.
    pub fn fingerprint(&self, asset_key: &str) -> Option<&str> {
        self.registry
            .get(asset_key)
            .map(|e| e.fingerprint_v0.as_str())
    }

    /// Look up fingerprint bytes for an asset_key.
    pub fn fingerprint_bytes(&self, asset_key: &str) -> Option<[u8; 32]> {
        self.registry
            .get(asset_key)
            .and_then(|e| hex_to_bytes(&e.fingerprint_v0))
    }

    /// Attach a verified physical CAS URI (`cas://sha256/<64_hex>`) to an already-materialized
    /// asset in the registry so downstream consumers and `verify_cache_hit` can verify its
    /// content-addressed backing blob (`ADR-0023` §2.2–2.3).
    pub fn bind_output_cas_uri(
        &mut self,
        asset_key: &str,
        cas_uri: &str,
        content_type: &str,
    ) -> io::Result<()> {
        let run_dir = self.sink.bundle().run_dir();
        verify_cas_uri_exists(run_dir, cas_uri, None)?;
        let entry = self.registry.get(asset_key).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("cannot bind CAS URI to unknown asset {asset_key}"),
            )
        })?;
        let mut updated = entry;
        updated.source = Some(SourceDescriptorV0 {
            uri: cas_uri.to_string(),
            content_type: content_type.to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: None,
        });
        self.sink.append_registry_update(&updated)?;
        self.registry.insert(asset_key.to_string(), updated);
        self.record_dataops_mutation()
    }

    /// Register a source dataset (no upstream; uses ingest_node for recipe_hash).
    pub fn register_source(
        &mut self,
        asset_key: &str,
        trust: TrustClass,
        source: SourceDescriptorV0,
        schema: Option<SchemaDescriptorV0>,
        ingest_node: &NodeV1,
    ) -> io::Result<()> {
        let source = sanitize_source_descriptor_v0(&source)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        let source_fp = source_fingerprint_v0(&source)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let schema_fp = schema
            .as_ref()
            .map(schema_hash_v0)
            .transpose()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?
            .unwrap_or(
                no_schema_hash_v0()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
            );

        // recipe_hash for source = hash(ingest_node_def, [])
        let recipe = recipe_hash_v0(ingest_node, &[])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let dataset_fp = dataset_fingerprint_v0(source_fp, schema_fp, recipe)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let entry = DatasetEntryV1 {
            asset_key: asset_key.to_string(),
            fingerprint_v0: hex_lower(&dataset_fp),
            source_fingerprint_v0: hex_lower(&source_fp),
            schema_hash_v0: hex_lower(&schema_fp),
            recipe_hash_v0: hex_lower(&recipe),
            trust,
            source: Some(source),
            schema,
            license_flags: Vec::new(),
            pii_tags: Vec::new(),
        };

        self.sink.append_registry_update(&entry)?;
        self.registry.insert(asset_key.to_string(), entry);
        self.record_dataops_mutation()
    }

    /// Materialize node outputs: derives fingerprints, propagates trust, emits records, flushes.
    pub fn materialize_node_outputs(
        &mut self,
        node: &NodeV1,
        outputs: &[OutputSpec],
        ts_unix_nanos: u64,
        cache_decision: impl Into<CacheDecisionV0>,
        duration_ms: u64,
    ) -> io::Result<()> {
        // ── PRE-VALIDATION ──────────────────────────────────────────────

        let mut provided: std::collections::HashSet<String> = std::collections::HashSet::new();
        {
            for output in outputs {
                if !provided.insert(output.asset_key.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("duplicate output asset_key: {}", output.asset_key),
                    ));
                }
            }
        }

        {
            let mut declared: std::collections::HashSet<String> = std::collections::HashSet::new();
            for node_output in &node.outputs {
                if !declared.insert(node_output.asset_key.clone()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "duplicate node output asset_key {} for node {}",
                            node_output.asset_key, node.node_key,
                        ),
                    ));
                }
            }
            for output_key in &provided {
                if !declared.contains(output_key) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "output {} not declared in node.outputs for node {}",
                            output_key, node.node_key,
                        ),
                    ));
                }
            }
            for declared_key in declared {
                if !provided.contains(&declared_key) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "missing declared node output {} for node {}",
                            declared_key, node.node_key,
                        ),
                    ));
                }
            }
        }

        // ── SNAPSHOT INPUTS (pre-mutation) ───────────────────────────────

        let mut upstream_fps: Vec<[u8; 32]> = Vec::new();
        let mut any_untrusted_input = false;
        let mut input_snapshots: Vec<(String, String)> = Vec::new();

        for input in &node.inputs {
            let entry = self.registry.get(&input.asset_key).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "missing input asset {} for node {}",
                        input.asset_key, node.node_key,
                    ),
                )
            })?;
            let fp_bytes = hex_to_bytes(&entry.fingerprint_v0).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid fingerprint for input asset {}: {}",
                        input.asset_key, entry.fingerprint_v0,
                    ),
                )
            })?;
            upstream_fps.push(fp_bytes);
            input_snapshots.push((input.asset_key.clone(), entry.fingerprint_v0.clone()));
            if matches!(entry.trust, TrustClass::Untrusted) {
                any_untrusted_input = true;
            }
        }

        // ── DERIVE + EMIT ───────────────────────────────────────────────

        let recipe = recipe_hash_v0(node, &upstream_fps)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let mut unsafe_reasons = Vec::new();
        if any_untrusted_input {
            unsafe_reasons.push(UnsafeReasonV0::UntrustedInput);
        }
        if !matches!(node.execution_trust, ExecutionTrust::Core) {
            unsafe_reasons.push(UnsafeReasonV0::UnsafeExtension);
        }
        let applied_transforms = self.pending_transform_audits.clone();
        if applied_transforms.iter().any(|audit| !audit.core_trusted)
            && !unsafe_reasons.contains(&UnsafeReasonV0::UnsafeExtension)
        {
            unsafe_reasons.push(UnsafeReasonV0::UnsafeExtension);
        }
        let unsafe_surface = !unsafe_reasons.is_empty();
        let output_trust = if unsafe_surface {
            TrustClass::Untrusted
        } else {
            TrustClass::Trusted
        };
        let cache_decision = cache_decision.into();
        let execution_profile = match node.execution_trust {
            ExecutionTrust::Core => "core",
            ExecutionTrust::SandboxedExtension => "sandboxed_extension",
            ExecutionTrust::UnsafeExtension => "unsafe_extension",
        };
        let cache_key = cache_key_v0(node, &upstream_fps, execution_profile)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let node_id = node
            .node_id
            .unwrap_or_else(|| node_id_from_key(&node.node_key));
        let node_id_str = node_id.to_string();
        let node_hash = hex_lower(
            &node_def_hash_v1(node)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
        );
        let input_asset_keys: Vec<String> = input_snapshots
            .iter()
            .map(|(asset_key, _)| asset_key.clone())
            .collect();
        let input_fingerprints_v0: Vec<String> =
            input_snapshots.iter().map(|(_, fp)| fp.clone()).collect();
        let mut staged_entries: Vec<DatasetEntryV1> = Vec::with_capacity(outputs.len());
        let mut staged_new_edges: Vec<((String, String, String), LineageEdgeV1)> = Vec::new();
        let mut staged_materializations: Vec<MaterializationRecordV2> =
            Vec::with_capacity(outputs.len());
        let mut staged_edge_keys: BTreeSet<(String, String, String)> = BTreeSet::new();
        let mut next_record_seq = self.next_record_seq;

        for output in outputs {
            let schema_fp = match output.schema.as_ref() {
                Some(s) => schema_hash_v0(s)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
                None => no_schema_hash_v0()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
            };

            let source_fp = derived_source_fingerprint_v0(&output.asset_key)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let dataset_fp = dataset_fingerprint_v0(source_fp, schema_fp, recipe)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let fp_hex = hex_lower(&dataset_fp);

            // Preserve existing source descriptor, schema, and physical metrics on verified cache hits.
            let existing_same_fp = self
                .registry
                .get(&output.asset_key)
                .filter(|existing| existing.fingerprint_v0 == fp_hex);
            let preserved_source = existing_same_fp.and_then(|existing| existing.source.clone());
            let preserved_schema = output
                .schema
                .clone()
                .or_else(|| existing_same_fp.and_then(|existing| existing.schema.clone()));
            let (cached_rows, cached_bytes) = if cache_decision == CacheDecisionV0::Hit {
                self.cached_output_metrics
                    .get(&(output.asset_key.clone(), fp_hex.clone()))
                    .copied()
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };
            let effective_rows = output.rows.or(cached_rows);
            let effective_bytes = output.bytes.or(cached_bytes);

            let entry = DatasetEntryV1 {
                asset_key: output.asset_key.clone(),
                fingerprint_v0: fp_hex.clone(),
                source_fingerprint_v0: hex_lower(&source_fp),
                schema_hash_v0: hex_lower(&schema_fp),
                recipe_hash_v0: hex_lower(&recipe),
                trust: output_trust,
                source: preserved_source,
                schema: preserved_schema,
                license_flags: Vec::new(),
                pii_tags: Vec::new(),
            };
            staged_entries.push(entry);

            for (_, in_fp) in &input_snapshots {
                let edge_key = (in_fp.clone(), fp_hex.clone(), node_id_str.clone());
                let edge = LineageEdgeV1 {
                    input_fingerprint_v0: in_fp.clone(),
                    output_fingerprint_v0: fp_hex.clone(),
                    node_id,
                    op_kind: node.op_kind,
                };
                if !self.lineage.contains_key(&edge_key)
                    && staged_edge_keys.insert(edge_key.clone())
                {
                    staged_new_edges.push((edge_key, edge));
                }
            }

            let mat = MaterializationRecordV2 {
                schema_version: MATERIALIZATION_SCHEMA_V2,
                record_seq: next_record_seq,
                ts_unix_nanos,
                asset_key: output.asset_key.clone(),
                fingerprint_v0: fp_hex,
                node_id,
                node_def_hash: node_hash.clone(),
                op_type: node.op_type.clone(),
                input_asset_keys: input_asset_keys.clone(),
                input_fingerprints_v0: input_fingerprints_v0.clone(),
                rows: effective_rows,
                bytes: effective_bytes,
                cache_decision,
                cache_reason: None,
                cache_key_v0: Some(cache_key.clone()),
                cache_hit: cache_hit_from_decision(cache_decision),
                duration_ms: Some(duration_ms),
                unsafe_surface,
                unsafe_reasons: unsafe_reasons.clone(),
                applied_transforms: applied_transforms.clone(),
                status: MaterializationStatusV0::Ok,
                error_code: None,
                quality: None,
            };
            staged_materializations.push(mat);
            next_record_seq = next_record_seq.saturating_add(1);
        }

        for entry in &staged_entries {
            self.sink.append_registry_update(entry)?;
        }
        for (_, edge) in &staged_new_edges {
            self.sink.append_lineage_edge_update(edge)?;
        }
        for mat in &staged_materializations {
            self.sink.append_materialization_v2(mat)?;
        }

        for mat in &staged_materializations {
            if mat.rows.is_some() || mat.bytes.is_some() {
                self.cached_output_metrics.insert(
                    (mat.asset_key.clone(), mat.fingerprint_v0.clone()),
                    (mat.rows, mat.bytes),
                );
            }
        }
        for entry in staged_entries {
            self.registry.insert(entry.asset_key.clone(), entry);
        }
        for (edge_key, edge) in staged_new_edges {
            self.lineage.insert(edge_key, edge);
        }
        self.next_record_seq = next_record_seq;
        self.pending_transform_audits.clear();

        self.record_dataops_mutation()
    }

    /// Compatibility wrapper for legacy call-sites that pass `cache_hit: bool`.
    pub fn materialize_node_outputs_cache_hit(
        &mut self,
        node: &NodeV1,
        outputs: &[OutputSpec],
        ts_unix_nanos: u64,
        cache_hit: bool,
        duration_ms: u64,
    ) -> io::Result<()> {
        self.materialize_node_outputs(
            node,
            outputs,
            ts_unix_nanos,
            CacheDecisionV0::from(cache_hit),
            duration_ms,
        )
    }

    fn record_dataops_mutation(&mut self) -> io::Result<()> {
        self.dataops_write_count = self.dataops_write_count.saturating_add(1);
        match self.snapshot_profile {
            SnapshotProfile::Strict => self.flush_snapshots(),
            SnapshotProfile::Streaming {
                snapshot_every_n_writes,
            } => {
                let period = snapshot_every_n_writes.max(1);
                if self.dataops_write_count % period == 0 {
                    self.flush_snapshots()?;
                }
                Ok(())
            }
        }
    }

    /// Flush registry.json + lineage.json atomically (crash-safe).
    fn flush_snapshots(&mut self) -> io::Result<()> {
        let registry = DatasetRegistryV1 {
            schema_version: DATAOPS_SCHEMA_V1,
            datasets: self.registry.values().cloned().collect(),
        };
        self.sink.write_dataset_registry(&registry)?;

        let lineage = DatasetLineageV1 {
            schema_version: DATAOPS_SCHEMA_V1,
            edges: self.lineage.values().cloned().collect(),
        };
        self.sink.write_dataset_lineage(&lineage)?;

        let run_dir = self.sink.bundle().run_dir();
        let datasets_dir = run_dir.join("datasets");
        let registry_sha256 = hex_lower(&sha256_file(&datasets_dir.join("registry.json"))?);
        let lineage_sha256 = hex_lower(&sha256_file(&datasets_dir.join("lineage.json"))?);
        let marker = SnapshotPairCommitV1 {
            schema_version: SNAPSHOT_PAIR_SCHEMA_V1,
            pair_seq: self.next_snapshot_pair_seq,
            registry_sha256,
            lineage_sha256,
        };
        write_json_pretty_atomic(&datasets_dir.join("snapshot_pair_commit.json"), &marker)?;
        self.next_snapshot_pair_seq = self.next_snapshot_pair_seq.saturating_add(1);

        if !matches!(
            self.sink.profile().manifest_policy,
            super::ManifestRefreshPolicy::FinalOnly
        ) {
            self.sink.finalize_manifest()?;
        }

        Ok(())
    }

    /// Finalize session: writes final snapshots and manifest.
    pub fn finalize(&mut self) -> io::Result<()> {
        self.flush_snapshots()?;
        self.sink.finalize_manifest()
    }

    /// Get a reference to the underlying sink for span/event/metric emission.
    pub fn sink(&self) -> &Arc<RunArtifactSink> {
        &self.sink
    }

    /// Snapshot clone of the current in-memory dataset registry.
    pub fn registry_snapshot(&self) -> DatasetRegistryV1 {
        DatasetRegistryV1 {
            schema_version: DATAOPS_SCHEMA_V1,
            datasets: self.registry.values().cloned().collect(),
        }
    }

    /// Resolve an input asset instance for scheduler execution.
    pub fn resolve_asset_instance(&self, asset_key: &str) -> Option<AssetInstanceV1> {
        self.registry.get(asset_key).map(|entry| AssetInstanceV1 {
            asset_key: entry.asset_key.clone(),
            fingerprint_v0: entry.fingerprint_v0.clone(),
            uri: entry.source.as_ref().map(|source| source.uri.clone()),
        })
    }

    /// Append `MaterializationStatusV0::Error` records for all declared node outputs.
    pub fn record_node_error(
        &mut self,
        node: &NodeV1,
        ts_unix_nanos: u64,
        duration_ms: u64,
        error_code: impl Into<String>,
    ) -> io::Result<()> {
        self.record_node_status(
            node,
            MaterializationStatusV0::Error,
            Some(error_code.into()),
            ts_unix_nanos,
            duration_ms,
        )
    }

    /// Append `MaterializationStatusV0::Skipped` records for all declared node outputs.
    pub fn record_node_skipped(
        &mut self,
        node: &NodeV1,
        ts_unix_nanos: u64,
        error_code: impl Into<String>,
    ) -> io::Result<()> {
        self.record_node_status(
            node,
            MaterializationStatusV0::Skipped,
            Some(error_code.into()),
            ts_unix_nanos,
            0,
        )
    }

    fn record_node_status(
        &mut self,
        node: &NodeV1,
        status: MaterializationStatusV0,
        error_code: Option<String>,
        ts_unix_nanos: u64,
        duration_ms: u64,
    ) -> io::Result<()> {
        let node_id = node
            .node_id
            .unwrap_or_else(|| node_id_from_key(&node.node_key));
        let node_hash = hex_lower(
            &node_def_hash_v1(node)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
        );

        let mut input_asset_keys = Vec::with_capacity(node.inputs.len());
        let mut input_fingerprints_v0 = Vec::with_capacity(node.inputs.len());
        let mut any_untrusted_input = false;
        let mut any_missing_input = false;
        for input in &node.inputs {
            input_asset_keys.push(input.asset_key.clone());
            if let Some(entry) = self.registry.get(&input.asset_key) {
                input_fingerprints_v0.push(entry.fingerprint_v0.clone());
                if matches!(entry.trust, TrustClass::Untrusted) {
                    any_untrusted_input = true;
                }
            } else {
                any_missing_input = true;
            }
        }

        let mut unsafe_reasons = Vec::new();
        if any_untrusted_input {
            unsafe_reasons.push(UnsafeReasonV0::UntrustedInput);
        }
        if !matches!(node.execution_trust, ExecutionTrust::Core) {
            unsafe_reasons.push(UnsafeReasonV0::UnsafeExtension);
        }
        if any_missing_input {
            unsafe_reasons.push(UnsafeReasonV0::MissingProvenance);
        }
        let unsafe_surface = !unsafe_reasons.is_empty();

        let mut staged = Vec::with_capacity(node.outputs.len());
        let mut next_record_seq = self.next_record_seq;
        for output in &node.outputs {
            let mut hasher = Sha256::new();
            hasher.update(b"swarmtorch.materialization.status.v0");
            hasher.update(node_id.as_bytes());
            hasher.update(output.asset_key.as_bytes());
            hasher.update(next_record_seq.to_be_bytes());
            hasher.update(match status {
                MaterializationStatusV0::Ok => b"ok".as_slice(),
                MaterializationStatusV0::Error => b"error".as_slice(),
                MaterializationStatusV0::Skipped => b"skipped".as_slice(),
            });
            let digest = hasher.finalize();
            let fingerprint_v0 = hex_lower(&digest);

            staged.push(MaterializationRecordV2 {
                schema_version: MATERIALIZATION_SCHEMA_V2,
                record_seq: next_record_seq,
                ts_unix_nanos,
                asset_key: output.asset_key.clone(),
                fingerprint_v0,
                node_id,
                node_def_hash: node_hash.clone(),
                op_type: node.op_type.clone(),
                input_asset_keys: input_asset_keys.clone(),
                input_fingerprints_v0: input_fingerprints_v0.clone(),
                rows: None,
                bytes: None,
                cache_decision: CacheDecisionV0::Unknown,
                cache_reason: None,
                cache_key_v0: None,
                cache_hit: None,
                duration_ms: Some(duration_ms),
                unsafe_surface,
                unsafe_reasons: unsafe_reasons.clone(),
                applied_transforms: Vec::new(),
                status,
                error_code: error_code.clone(),
                quality: None,
            });
            next_record_seq = next_record_seq.saturating_add(1);
        }

        for record in &staged {
            self.sink.append_materialization_v2(record)?;
        }
        self.next_record_seq = next_record_seq;
        Ok(())
    }

    // ── Cache-hit prediction & CAS verification (alpha.6+ / Wave 10) ──

    /// Predict output fingerprints for a node before execution.
    pub fn predict(
        &self,
        node: &NodeV1,
        outputs: &[OutputSpecCore],
    ) -> Result<Vec<PredictedOutput>, PredictError> {
        let mut provided: std::collections::HashSet<String> = std::collections::HashSet::new();
        for output in outputs {
            if !provided.insert(output.asset_key.clone()) {
                return Err(PredictError::OutputContract(format!(
                    "duplicate output asset_key {}",
                    output.asset_key
                )));
            }
        }

        let mut declared: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node_output in &node.outputs {
            if !declared.insert(node_output.asset_key.clone()) {
                return Err(PredictError::OutputContract(format!(
                    "duplicate node output asset_key {} for node {}",
                    node_output.asset_key, node.node_key,
                )));
            }
        }
        for output_key in &provided {
            if !declared.contains(output_key) {
                return Err(PredictError::OutputContract(format!(
                    "output {} not declared in node.outputs for node {}",
                    output_key, node.node_key,
                )));
            }
        }
        for declared_key in declared {
            if !provided.contains(&declared_key) {
                return Err(PredictError::OutputContract(format!(
                    "missing declared node output {} for node {}",
                    declared_key, node.node_key,
                )));
            }
        }

        let mut upstream_fps: Vec<[u8; 32]> = Vec::new();

        for input in &node.inputs {
            let entry = self
                .registry
                .get(&input.asset_key)
                .ok_or_else(|| PredictError::MissingInput(input.asset_key.clone()))?;
            let fp_bytes = hex_to_bytes(&entry.fingerprint_v0)
                .ok_or_else(|| PredictError::InvalidFingerprint(input.asset_key.clone()))?;
            upstream_fps.push(fp_bytes);
        }

        let predicted = predict_output_fingerprints(node, outputs, &upstream_fps)
            .map_err(|e| PredictError::InvalidFingerprint(e.to_string()))?;
        Ok(predicted)
    }

    /// Asset-key scoped cache hit check.
    pub fn is_cache_hit(&self, asset_key: &str, predicted_fp: &str) -> bool {
        self.registry
            .get(asset_key)
            .map(|e| e.fingerprint_v0 == predicted_fp)
            .unwrap_or(false)
    }

    /// Return the declared `content_type` for `asset_key` if registered with a [`SourceDescriptorV0`].
    pub fn asset_content_type(&self, asset_key: &str) -> Option<&str> {
        self.registry
            .get(asset_key)
            .and_then(|entry| entry.source.as_ref())
            .map(|source| source.content_type.as_str())
    }

    /// Return cached schema, row count, and byte count for a registered `asset_key`.
    pub fn cached_output_spec(&self, asset_key: &str) -> Option<OutputSpec> {
        let entry = self.registry.get(asset_key)?;
        let (rows, bytes) = self
            .cached_output_metrics
            .get(&(asset_key.to_string(), entry.fingerprint_v0.clone()))
            .copied()
            .unwrap_or((None, None));
        Some(OutputSpec {
            asset_key: asset_key.to_string(),
            schema: entry.schema.clone(),
            rows,
            bytes,
        })
    }

    /// Evaluate whether `node` can reuse all of its declared `outputs` from cache
    /// (`ADR-0023` §2.3, `W10-01`).
    ///
    /// - Respects [`CachePolicyV0::Bypass`] by returning [`CacheDecisionV0::Bypass`].
    /// - Requires `node.outputs` to be non-empty and every predicted output to match the current
    ///   registry entry's `fingerprint_v0`.
    /// - Returns `Ok(CacheDecisionV0::Miss)` when no matching cached output is registered.
    /// - When a recorded cache entry matches the predicted fingerprint and binds a
    ///   `cas://sha256/...` or `blobs/sha256/...` URI, verifies that the physical CAS blob exists
    ///   and its SHA-256 matches, returning `Err(io::Error)` on missing/corrupt CAS integrity failure
    ///   instead of silently downgrading to `Miss`.
    pub fn verify_cache_hit(
        &self,
        node: &NodeV1,
        outputs: &[OutputSpecCore],
    ) -> io::Result<CacheDecisionV0> {
        if matches!(node.cache_policy, Some(CachePolicyV0::Bypass)) {
            return Ok(CacheDecisionV0::Bypass);
        }
        if outputs.is_empty() {
            return Ok(CacheDecisionV0::Miss);
        }
        let Ok(predicted) = self.predict(node, outputs) else {
            return Ok(CacheDecisionV0::Miss);
        };
        let run_dir = self.sink.bundle().run_dir();
        for pred in &predicted {
            let Some(entry) = self.registry.get(&pred.asset_key) else {
                return Ok(CacheDecisionV0::Miss);
            };
            if entry.fingerprint_v0 != pred.fingerprint_v0 {
                return Ok(CacheDecisionV0::Miss);
            }
            if let Some(ref src) = entry.source {
                if parse_cas_uri_digest(&src.uri).is_some() {
                    verify_cas_uri_exists(run_dir, &src.uri, None)?;
                }
            }
        }
        Ok(CacheDecisionV0::Hit)
    }

    #[cfg(test)]
    pub(crate) fn test_registry_entry_mut(
        &mut self,
        asset_key: &str,
    ) -> Option<&mut DatasetEntryV1> {
        self.registry.get_mut(asset_key)
    }
}

fn hex_to_bytes(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let high = hex_digit(chunk[0])?;
        let low = hex_digit(chunk[1])?;
        out[i] = (high << 4) | low;
    }
    Some(out)
}

fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}
