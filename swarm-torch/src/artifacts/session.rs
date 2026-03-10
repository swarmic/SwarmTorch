use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use swarm_torch_core::dataops::{
    cache_hit_from_decision, cache_key_v0, dataset_fingerprint_v0, derived_source_fingerprint_v0,
    no_schema_hash_v0, predict_output_fingerprints, recipe_hash_v0, sanitize_source_descriptor_v0,
    schema_hash_v0, source_fingerprint_v0, CacheDecisionV0, DatasetEntryV1, DatasetLineageV1,
    DatasetRegistryV1, LineageEdgeV1, MaterializationRecordV2, MaterializationStatusV0,
    OutputSpecCore, PredictedOutput, SchemaDescriptorV0, SourceDescriptorV0, TransformAuditV0,
    TrustClass, UnsafeReasonV0, DATAOPS_SCHEMA_V1, MATERIALIZATION_SCHEMA_V2,
};
use swarm_torch_core::execution::AssetInstanceV1;
use swarm_torch_core::run_graph::{node_def_hash_v1, node_id_from_key, ExecutionTrust, NodeV1};

use super::io::{hex_lower, sha256_file, write_json_pretty_atomic};
use super::{RunArtifactSink, SnapshotProfile};

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

const SNAPSHOT_PAIR_SCHEMA_V1: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SnapshotPairCommitV1 {
    schema_version: u32,
    pair_seq: u64,
    registry_sha256: String,
    lineage_sha256: String,
}

/// DataOps session: manages registry/lineage with trust propagation and crash-safe persistence.
///
/// **Limitation (v0.1):** Single-process writer per run directory.
/// The `RunArtifactSink` mutex is in-process only; concurrent processes writing to the
/// same bundle will corrupt NDJSON files.
///
/// **Manifest behavior:** manifest freshness is controlled by `ArtifactWriteProfile`.
/// With `ManifestRefreshPolicy::FinalOnly`, call `finalize()` before reading the bundle
/// with report tools (which validate manifest hashes). `Always` and `IntervalN`
/// can keep manifests fresh mid-run at the cost of extra hashing I/O.
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
        }
    }

    /// Record an applied update transform for the next materialization emission.
    ///
    /// These audits are attached to the next `materialize_node_outputs` call and then cleared.
    pub fn record_transform_applied(&mut self, audit: &TransformAuditV0) {
        self.pending_transform_audits.push(audit.clone());
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

        self.registry.insert(asset_key.to_string(), entry.clone());
        self.sink.append_registry_update(&entry)?;
        self.record_dataops_mutation()
    }

    /// Materialize node outputs: derives fingerprints, propagates trust, emits records, flushes.
    ///
    /// **Correctness guarantees (alpha.6+):**
    /// - Returns `Err` if any `node.inputs[].asset_key` is missing from registry (fail closed)
    /// - Returns `Err` if any input fingerprint is invalid hex
    /// - Returns `Err` if any `OutputSpec.asset_key` is not declared in `node.outputs[]`
    /// - Returns `Err` if any declared `node.outputs[]` asset key is missing from `outputs`
    /// - Returns `Err` if `node.outputs[]` itself contains duplicate `asset_key` values
    /// - Returns `Err` if `outputs` contains duplicate `asset_key` values
    /// - Lineage edges reference pre-mutation input fingerprints (not post-insert state)
    ///
    /// Safety taxonomy:
    /// - `unsafe_reasons` includes `UntrustedInput` when any input trust is untrusted.
    /// - `unsafe_reasons` includes `UnsafeExtension` when `execution_trust != Core`.
    /// - `unsafe_surface` is derived from reasons (`!unsafe_reasons.is_empty()`).
    pub fn materialize_node_outputs(
        &mut self,
        node: &NodeV1,
        outputs: &[OutputSpec],
        ts_unix_nanos: u64,
        cache_decision: impl Into<CacheDecisionV0>,
        duration_ms: u64,
    ) -> io::Result<()> {
        // ── PRE-VALIDATION ──────────────────────────────────────────────

        // 1. Reject duplicate output keys, and capture provided output set.
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

        // 2. Enforce output contract:
        //    - every provided OutputSpec must be declared in node.outputs[]
        //    - every declared node output must be present in outputs
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

        // 3. Fail closed: every declared input MUST exist with a valid fingerprint.
        //    Capture snapshots before any registry mutation.
        let mut upstream_fps: Vec<[u8; 32]> = Vec::new();
        let mut any_untrusted_input = false;
        let mut input_snapshots: Vec<(String, String)> = Vec::new(); // (asset_key, fp_hex)

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

        // 4. Compute recipe_hash_v0(node, upstream_fps)
        let recipe = recipe_hash_v0(node, &upstream_fps)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        // 5. Derive unsafe reasons and output trust classification.
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

        // Derive node_id
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

        // 6. For each output: compute fingerprint, staged entry/lineage/materialization.
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

            let entry = DatasetEntryV1 {
                asset_key: output.asset_key.clone(),
                fingerprint_v0: fp_hex.clone(),
                source_fingerprint_v0: hex_lower(&source_fp),
                schema_hash_v0: hex_lower(&schema_fp),
                recipe_hash_v0: hex_lower(&recipe),
                trust: output_trust,
                source: None,
                schema: output.schema.clone(),
                license_flags: Vec::new(),
                pii_tags: Vec::new(),
            };
            staged_entries.push(entry);

            // 7. Lineage edges from pre-mutation input snapshots (not live registry)
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

            // 8. Append MaterializationRecordV2
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
                rows: output.rows,
                bytes: output.bytes,
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

        // 9. Persist staged writes before mutating in-memory state.
        for entry in &staged_entries {
            self.sink.append_registry_update(entry)?;
        }
        for (_, edge) in &staged_new_edges {
            self.sink.append_lineage_edge_update(edge)?;
        }
        for mat in &staged_materializations {
            self.sink.append_materialization_v2(mat)?;
        }

        // 10. Commit in-memory state only after all writes succeed.
        for entry in staged_entries {
            self.registry.insert(entry.asset_key.clone(), entry);
        }
        for (edge_key, edge) in staged_new_edges {
            self.lineage.insert(edge_key, edge);
        }
        self.next_record_seq = next_record_seq;
        self.pending_transform_audits.clear();

        // 11. Snapshot compaction (strict or streaming cadence).
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
        // Registry: sorted by asset_key (BTreeMap iteration is already sorted)
        let registry = DatasetRegistryV1 {
            schema_version: DATAOPS_SCHEMA_V1,
            datasets: self.registry.values().cloned().collect(),
        };
        self.sink.write_dataset_registry(&registry)?;

        // Lineage: sorted by key (BTreeMap iteration is already sorted)
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

        // Keep manifest freshness aligned when snapshot pair markers change.
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

    // ── Cache-hit prediction (alpha.6+) ─────────────────────────────

    /// Predict output fingerprints for a node before execution.
    ///
    /// **Fail closed:** returns `Err(PredictError::MissingInput)` if any
    /// `node.inputs[].asset_key` is absent from the registry, and
    /// `Err(PredictError::InvalidFingerprint)` if a registered fingerprint
    /// is malformed.
    ///
    /// **Output contract enforcement:** returns `Err(PredictError::OutputContract)` if
    /// `outputs` violates node output declarations (undeclared/missing/duplicate keys).
    ///
    /// The orchestrator must treat errors as "no cache optimization".
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
    ///
    /// Returns `true` iff `asset_key` exists in the registry **and** its
    /// current fingerprint matches `predicted_fp`. This prevents false positives
    /// from fingerprint collisions across different asset keys.
    pub fn is_cache_hit(&self, asset_key: &str, predicted_fp: &str) -> bool {
        self.registry
            .get(asset_key)
            .map(|e| e.fingerprint_v0 == predicted_fp)
            .unwrap_or(false)
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
