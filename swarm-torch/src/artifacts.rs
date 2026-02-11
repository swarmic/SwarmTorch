//! Run artifact bundle writer/validator (std-only).
//!
//! This implements the on-disk "artifact spine" described in ADR-0016:
//! `runs/<run_id>/...` with a path-addressed SHA-256 `manifest.json` and
//! NDJSON baselines for spans/events/metrics/materializations.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use swarm_torch_core::dataops::{DatasetLineageV1, DatasetRegistryV1, MaterializationRecordV1};
use swarm_torch_core::observe::{EventRecord, MetricRecord, RunId, SpanRecord};
use swarm_torch_core::run_graph::GraphV1;

const SCHEMA_VERSION_V1: u32 = 1;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ManifestV1 {
    schema_version: u32,
    run_id: RunId,
    hash_algo: String,
    entries: Vec<ManifestEntryV1>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ManifestEntryV1 {
    // Path relative to `runs/<run_id>/`.
    path: String,
    sha256: String, // lowercase hex
    bytes: u64,
    required: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct RunFileV1 {
    schema_version: u32,
    run_id: RunId,
    created_unix_nanos: u64,
    swarmtorch_version: String,
}

/// A writer/validator for a single run artifact bundle (`runs/<run_id>/...`).
#[derive(Debug, Clone)]
pub struct RunArtifactBundle {
    run_dir: PathBuf,
    run_id: RunId,
}

/// Thread-safe artifact sink (single-writer enforced by an in-process mutex).
///
/// This is the simplest v0.1 strategy for multi-producer telemetry without risking
/// interleaved NDJSON lines.
#[derive(Debug)]
pub struct RunArtifactSink {
    bundle: RunArtifactBundle,
    lock: Mutex<()>,
}

impl swarm_torch_core::observe::RunEventEmitter for RunArtifactSink {
    type Error = io::Error;

    fn emit_span(&self, span: &SpanRecord) -> Result<(), Self::Error> {
        self.append_span(span)
    }

    fn emit_event(&self, event: &EventRecord) -> Result<(), Self::Error> {
        self.append_event(event)
    }

    fn emit_metric(&self, metric: &MetricRecord) -> Result<(), Self::Error> {
        self.append_metric(metric)
    }
}

impl RunArtifactSink {
    pub fn new(bundle: RunArtifactBundle) -> Self {
        Self {
            bundle,
            lock: Mutex::new(()),
        }
    }

    pub fn bundle(&self) -> &RunArtifactBundle {
        &self.bundle
    }

    fn guard(&self) -> io::Result<std::sync::MutexGuard<'_, ()>> {
        self.lock
            .lock()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "artifact sink mutex poisoned"))
    }

    pub fn write_graph(&self, graph: &GraphV1) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.write_graph(graph)
    }

    pub fn append_span(&self, span: &SpanRecord) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.append_span(span)
    }

    pub fn append_event(&self, event: &EventRecord) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.append_event(event)
    }

    pub fn append_metric(&self, metric: &MetricRecord) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.append_metric(metric)
    }

    pub fn append_materialization(&self, m: &MaterializationRecordV1) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.append_materialization(m)
    }

    pub fn write_dataset_registry(&self, r: &DatasetRegistryV1) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.write_dataset_registry(r)
    }

    pub fn write_dataset_lineage(&self, l: &DatasetLineageV1) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.write_dataset_lineage(l)
    }

    pub fn finalize_manifest(&self) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.finalize_manifest()
    }

    pub fn validate_manifest(&self) -> io::Result<()> {
        let _g = self.guard()?;
        self.bundle.validate_manifest()
    }
}

// ---------------------------------------------------------------------------
// DataOpsSession: single derivation point for fingerprints/lineage/materializations
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;
use std::sync::Arc;

use swarm_torch_core::dataops::{
    dataset_fingerprint_v0, derived_source_fingerprint_v0, no_schema_hash_v0, recipe_hash_v0,
    schema_hash_v0, source_fingerprint_v0, DatasetEntryV1, LineageEdgeV1, SchemaDescriptorV0,
    SourceDescriptorV0, TrustClass, DATAOPS_SCHEMA_V1,
};
use swarm_torch_core::run_graph::{node_def_hash_v1, node_id_from_key, ExecutionTrust, NodeV1};

/// Output specification for `materialize_node_outputs`.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub asset_key: String,
    pub schema: Option<SchemaDescriptorV0>,
    pub rows: Option<u64>,
    pub bytes: Option<u64>,
}

/// DataOps session: manages registry/lineage with trust propagation and crash-safe persistence.
///
/// **Limitation (v0.1):** Single-process writer per run directory.
/// The `RunArtifactSink` mutex is in-process only; concurrent processes writing to the
/// same bundle will corrupt NDJSON files.
///
/// **Manifest gap:** `flush_snapshots()` writes registry.json/lineage.json after each
/// materialization but does NOT update manifest.json. Call `finalize()` before reading
/// the bundle with report tools (which validate manifest hashes). After a crash mid-session,
/// the manifest will be stale and report generation will fail until `finalize()` is called.
#[derive(Debug)]
pub struct DataOpsSession {
    sink: Arc<RunArtifactSink>,
    /// asset_key -> DatasetEntryV1 (uniqueness enforced)
    registry: BTreeMap<String, DatasetEntryV1>,
    /// (input_fp, output_fp, node_id_str) -> LineageEdgeV1 (dedupe key)
    lineage: BTreeMap<(String, String, String), LineageEdgeV1>,
}

impl DataOpsSession {
    /// Create a new session wrapping an artifact sink.
    pub fn new(sink: Arc<RunArtifactSink>) -> Self {
        Self {
            sink,
            registry: BTreeMap::new(),
            lineage: BTreeMap::new(),
        }
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
        let source_fp = source_fingerprint_v0(&source);
        let schema_fp = schema
            .as_ref()
            .map(schema_hash_v0)
            .unwrap_or_else(no_schema_hash_v0);

        // recipe_hash for source = hash(ingest_node_def, [])
        let recipe = recipe_hash_v0(ingest_node, &[]);

        let dataset_fp = dataset_fingerprint_v0(source_fp, schema_fp, recipe);

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

        self.registry.insert(asset_key.to_string(), entry);
        self.flush_snapshots()
    }

    /// Materialize node outputs: derives fingerprints, propagates trust, emits records, flushes.
    ///
    /// **Correctness guarantees (alpha.6+):**
    /// - Returns `Err` if any `node.inputs[].asset_key` is missing from registry (fail closed)
    /// - Returns `Err` if any input fingerprint is invalid hex
    /// - Returns `Err` if any `OutputSpec.asset_key` is not declared in `node.outputs[]`
    /// - Returns `Err` if `outputs` contains duplicate `asset_key` values
    /// - Lineage edges reference pre-mutation input fingerprints (not post-insert state)
    ///
    /// Sets `unsafe_surface = true` if:
    /// - `node.execution_trust != Core`, OR
    /// - any input has `trust = Untrusted`
    pub fn materialize_node_outputs(
        &mut self,
        node: &NodeV1,
        outputs: &[OutputSpec],
        ts_unix_nanos: u64,
        cache_hit: bool,
        duration_ms: u64,
    ) -> io::Result<()> {
        // ── PRE-VALIDATION ──────────────────────────────────────────────

        // 1. Reject duplicate output keys
        {
            let mut seen = std::collections::HashSet::new();
            for output in outputs {
                if !seen.insert(&output.asset_key) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("duplicate output asset_key: {}", output.asset_key),
                    ));
                }
            }
        }

        // 2. Enforce output contract: every OutputSpec must be declared in node.outputs[]
        {
            let declared: std::collections::HashSet<&str> =
                node.outputs.iter().map(|o| o.asset_key.as_str()).collect();
            for output in outputs {
                if !declared.contains(output.asset_key.as_str()) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "output {} not declared in node.outputs for node {}",
                            output.asset_key, node.node_key,
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
        let recipe = recipe_hash_v0(node, &upstream_fps);

        // 5. Determine output trust
        let output_trust =
            if any_untrusted_input || !matches!(node.execution_trust, ExecutionTrust::Core) {
                TrustClass::Untrusted
            } else {
                TrustClass::Trusted
            };

        let unsafe_surface = matches!(output_trust, TrustClass::Untrusted);

        // Derive node_id
        let node_id = node
            .node_id
            .unwrap_or_else(|| node_id_from_key(&node.node_key));
        let node_id_str = node_id.to_string();
        let node_hash = hex_lower(&node_def_hash_v1(node));

        // 6. For each output: compute fingerprint, insert entry, create lineage, emit record
        for output in outputs {
            let schema_fp = output
                .schema
                .as_ref()
                .map(schema_hash_v0)
                .unwrap_or_else(no_schema_hash_v0);

            let source_fp = derived_source_fingerprint_v0(&output.asset_key);
            let dataset_fp = dataset_fingerprint_v0(source_fp, schema_fp, recipe);
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
            self.registry.insert(output.asset_key.clone(), entry);

            // 7. Lineage edges from pre-mutation input snapshots (not live registry)
            for (_, in_fp) in &input_snapshots {
                let edge_key = (in_fp.clone(), fp_hex.clone(), node_id_str.clone());
                let edge = LineageEdgeV1 {
                    input_fingerprint_v0: in_fp.clone(),
                    output_fingerprint_v0: fp_hex.clone(),
                    node_id,
                    op_kind: node.op_kind,
                };
                self.lineage.insert(edge_key, edge);
            }

            // 8. Append MaterializationRecordV1
            let mat = MaterializationRecordV1 {
                schema_version: DATAOPS_SCHEMA_V1,
                ts_unix_nanos,
                asset_key: output.asset_key.clone(),
                fingerprint_v0: fp_hex,
                node_id,
                node_def_hash: node_hash.clone(),
                rows: output.rows,
                bytes: output.bytes,
                cache_hit: Some(cache_hit),
                duration_ms: Some(duration_ms),
                quality_flags: None,
                unsafe_surface,
            };
            self.sink.append_materialization(&mat)?;
        }

        // 9. flush_snapshots()
        self.flush_snapshots()
    }

    /// Flush registry.json + lineage.json atomically (crash-safe).
    fn flush_snapshots(&self) -> io::Result<()> {
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

        Ok(())
    }

    /// Finalize session: writes final snapshots and manifest.
    pub fn finalize(&self) -> io::Result<()> {
        self.flush_snapshots()?;
        self.sink.finalize_manifest()
    }

    /// Get a reference to the underlying sink for span/event/metric emission.
    pub fn sink(&self) -> &Arc<RunArtifactSink> {
        &self.sink
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

        let mut graph = GraphV1::default();
        graph.graph_id = Some(run_id.to_string());
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

        let bundle = Self { run_dir, run_id };
        // Emit an initial manifest so a bundle is valid immediately.
        bundle.finalize_manifest()?;
        Ok(bundle)
    }

    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Write (replace) `graph.json` with a normalized graph.
    ///
    /// This computes derived fields (`node_id`, `node_def_hash`) according to ADR-0017.
    pub fn write_graph(&self, graph: &GraphV1) -> io::Result<()> {
        let mut g = graph.clone();
        for node in &mut g.nodes {
            if node.code_ref.as_deref().unwrap_or("").is_empty() {
                node.code_ref = Some(format!("swarm-torch@{}", env!("CARGO_PKG_VERSION")));
            }
        }
        let normalized = g.normalize();
        write_json_pretty_atomic(&self.run_dir.join("graph.json"), &normalized)
    }

    pub fn append_span(&self, span: &SpanRecord) -> io::Result<()> {
        append_ndjson(&self.run_dir.join("spans.ndjson"), span)
    }

    pub fn append_event(&self, event: &EventRecord) -> io::Result<()> {
        append_ndjson(&self.run_dir.join("events.ndjson"), event)
    }

    pub fn append_metric(&self, metric: &MetricRecord) -> io::Result<()> {
        append_ndjson(&self.run_dir.join("metrics.ndjson"), metric)
    }

    pub fn append_materialization(
        &self,
        materialization: &MaterializationRecordV1,
    ) -> io::Result<()> {
        append_ndjson(
            &self
                .run_dir
                .join("datasets")
                .join("materializations.ndjson"),
            materialization,
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

    /// Optional durability hook: fsync all required v1 files (best-effort).
    ///
    /// This is intentionally not called by default for performance reasons.
    pub fn sync_required_v1(&self) -> io::Result<()> {
        for rel in required_paths_v1() {
            let path = self.run_dir.join(rel);
            let f = File::open(&path)?;
            // Best-effort: ignore sync errors on platforms/filesystems that don't support it.
            let _ = f.sync_all();
        }
        Ok(())
    }

    /// (Re)compute and write `manifest.json` for all current files in the bundle.
    ///
    /// Note: `manifest.json` is excluded from itself (non-self-referential).
    pub fn finalize_manifest(&self) -> io::Result<()> {
        // Ensure baseline v1 required files exist before hashing.
        for p in required_paths_v1() {
            let full = self.run_dir.join(p);
            if !full.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing required bundle file: {p}"),
                ));
            }
        }

        let mut files = Vec::new();
        collect_files_recursive(&self.run_dir, &mut files)?;

        let mut entries = Vec::new();
        for file_path in files {
            if file_path.file_name().and_then(|s| s.to_str()) == Some("manifest.json") {
                continue;
            }
            let rel = rel_path_string(&file_path, &self.run_dir)?;
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

        let manifest = ManifestV1 {
            schema_version: SCHEMA_VERSION_V1,
            run_id: self.run_id,
            hash_algo: "sha256".to_string(),
            entries,
        };

        write_json_pretty_atomic(&self.run_dir.join("manifest.json"), &manifest)
    }

    /// Validate `manifest.json` against current on-disk bytes.
    pub fn validate_manifest(&self) -> io::Result<()> {
        let manifest_path = self.run_dir.join("manifest.json");
        let manifest: ManifestV1 = read_json(&manifest_path)?;

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

        for entry in manifest.entries {
            let path = self.run_dir.join(&entry.path);
            if !path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing file listed in manifest: {}", entry.path),
                ));
            }
            let meta = fs::metadata(&path)?;
            if meta.len() != entry.bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "size mismatch for {} (expected {})",
                        entry.path, entry.bytes
                    ),
                ));
            }
            let digest = sha256_file(&path)?;
            let actual = hex_lower(&digest);
            if actual != entry.sha256 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("sha256 mismatch for {}", entry.path),
                ));
            }
        }

        Ok(())
    }
}

fn ensure_file(path: &Path) -> io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    File::create(path)?;
    Ok(())
}

fn write_json_pretty_atomic<T: serde::Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let json =
        serde_json::to_vec_pretty(value).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    atomic_write(path, &json)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> io::Result<T> {
    let file = File::open(path)?;
    serde_json::from_reader(file).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

fn append_ndjson<T: serde::Serialize>(path: &Path, record: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let line =
        serde_json::to_string(record).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let mut buf = line.into_bytes();
    buf.push(b'\n');
    file.write_all(&buf)?;
    file.flush()?;
    Ok(())
}

fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            collect_files_recursive(&path, out)?;
        } else if ty.is_file() {
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.ends_with(".tmp"))
                .unwrap_or(false)
            {
                continue;
            }
            out.push(path);
        }
    }
    Ok(())
}

fn rel_path_string(path: &Path, base: &Path) -> io::Result<String> {
    let rel = path.strip_prefix(base).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "path is not within bundle root",
        )
    })?;

    let mut parts = Vec::new();
    for c in rel.components() {
        parts.push(c.as_os_str().to_string_lossy().to_string());
    }
    Ok(parts.join("/"))
}

fn sha256_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..]);
    Ok(out)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid file name"))?;

    let tmp_name = format!("{file_name}.tmp");
    let tmp_path = path.with_file_name(tmp_name);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(bytes)?;
        f.flush()?;
        // Not a hard durability guarantee, but improves crash-safety for small files.
        let _ = f.sync_all();
    }

    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn required_paths_v1() -> &'static [&'static str] {
    &[
        "run.json",
        "graph.json",
        "spans.ndjson",
        "events.ndjson",
        "metrics.ndjson",
        "datasets/registry.json",
        "datasets/lineage.json",
        "datasets/materializations.ndjson",
    ]
}

fn is_required_path_v1(p: &str) -> bool {
    required_paths_v1().iter().any(|rp| *rp == p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::dataops::MaterializationRecordV1;
    use swarm_torch_core::observe::{AttrMap, SpanId, TraceId};
    use swarm_torch_core::run_graph::{
        node_def_hash_v1, node_id_from_key, AssetRefV1, CanonParams, CanonValue, ExecutionTrust,
        GraphV1, NodeV1, OpKind,
    };

    fn temp_dir(prefix: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("swarmtorch_{prefix}_{pid}_{nanos}"));
        dir
    }

    #[test]
    fn bundle_manifest_roundtrip() {
        let base = temp_dir("bundle_manifest_roundtrip");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([1u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

        // Append at least one span record.
        let span = SpanRecord {
            schema_version: 1,
            trace_id: TraceId::from_bytes([2u8; 16]),
            span_id: SpanId::from_bytes([3u8; 8]),
            parent_span_id: None,
            name: "test".to_string(),
            start_unix_nanos: 1,
            end_unix_nanos: Some(2),
            attrs: AttrMap::new(),
        };
        bundle.append_span(&span).unwrap();

        // Append at least one materialization record.
        let m = MaterializationRecordV1 {
            schema_version: 1,
            ts_unix_nanos: 1,
            asset_key: "dataset://ns/users_clean".to_string(),
            fingerprint_v0: "deadbeef".to_string(),
            node_id: node_id_from_key("prep/clean_users"),
            node_def_hash: "00".repeat(32),
            rows: None,
            bytes: None,
            cache_hit: None,
            duration_ms: None,
            quality_flags: None,
            unsafe_surface: false,
        };
        bundle.append_materialization(&m).unwrap();

        bundle.finalize_manifest().unwrap();
        bundle.validate_manifest().unwrap();

        // Cleanup.
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn graph_write_normalizes_ids_and_hashes() {
        let base = temp_dir("graph_write_normalizes");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([9u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

        let mut graph = GraphV1::default();
        graph.nodes.push(NodeV1 {
            node_key: "prep/clean_users".to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "validate".to_string(),
            inputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/users".to_string(),
                fingerprint: None,
            }],
            outputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/users_clean".to_string(),
                fingerprint: None,
            }],
            params: CanonParams::new(),
            code_ref: None,
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
        });

        bundle.write_graph(&graph).unwrap();

        let on_disk: GraphV1 = read_json(&bundle.run_dir.join("graph.json")).unwrap();
        assert_eq!(on_disk.schema_version, 1);
        assert_eq!(on_disk.nodes.len(), 1);

        let node = &on_disk.nodes[0];
        assert!(node.node_id.is_some());
        assert!(node.node_def_hash.is_some());
        assert!(node.code_ref.is_some());

        // Verify derived values match the core hashing rules.
        let expected_id = node_id_from_key(&node.node_key).to_string();
        assert_eq!(node.node_id.unwrap().to_string(), expected_id);

        let digest = node_def_hash_v1(node);
        let expected_hash = hex_lower(&digest);
        assert_eq!(node.node_def_hash.as_ref().unwrap(), &expected_hash);

        let _ = fs::remove_dir_all(&base);
    }

    // -------------------------------------------------------------------------
    // DataOpsSession invariant tests
    // -------------------------------------------------------------------------

    fn make_source_node(key: &str) -> NodeV1 {
        NodeV1 {
            node_key: key.to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "ingest".to_string(),
            inputs: vec![],
            outputs: vec![],
            params: CanonParams::new(),
            code_ref: Some("test@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
        }
    }

    fn make_transform_node(
        key: &str,
        input_keys: &[&str],
        output_keys: &[&str],
        trust: ExecutionTrust,
    ) -> NodeV1 {
        NodeV1 {
            node_key: key.to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "transform".to_string(),
            inputs: input_keys
                .iter()
                .map(|k| AssetRefV1 {
                    asset_key: k.to_string(),
                    fingerprint: None,
                })
                .collect(),
            outputs: output_keys
                .iter()
                .map(|k| AssetRefV1 {
                    asset_key: k.to_string(),
                    fingerprint: None,
                })
                .collect(),
            params: CanonParams::new(),
            code_ref: Some("test@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: trust,
            node_def_hash: None,
        }
    }

    #[test]
    fn fingerprint_changes_when_node_def_changes() {
        let base = temp_dir("fp_changes_node_def");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([10u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register source
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest1 = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source.clone(),
                None,
                &ingest1,
            )
            .unwrap();
        let fp1 = session.fingerprint("dataset://ns/raw").unwrap().to_string();

        // Register same source with different ingest node (different params)
        let mut ingest2 = make_source_node("ingest/v2");
        ingest2
            .params
            .insert("delimiter".to_string(), CanonValue::Str(",".to_string()));

        // Create new session to simulate different parse options
        let bundle2 = RunArtifactBundle::create(
            temp_dir("fp_changes_node_def2"),
            RunId::from_bytes([11u8; 16]),
        )
        .unwrap();
        let sink2 = Arc::new(RunArtifactSink::new(bundle2));
        let mut session2 = DataOpsSession::new(Arc::clone(&sink2));
        session2
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source,
                None,
                &ingest2,
            )
            .unwrap();
        let fp2 = session2
            .fingerprint("dataset://ns/raw")
            .unwrap()
            .to_string();

        assert_ne!(
            fp1, fp2,
            "fingerprint should change when ingest node changes"
        );
        assert_eq!(fp1.len(), 64, "fingerprint should be 64-char hex");
        assert_eq!(fp2.len(), 64, "fingerprint should be 64-char hex");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn fingerprint_changes_when_upstream_changes() {
        let base = temp_dir("fp_changes_upstream");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([20u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register two different sources
        let source1 = SourceDescriptorV0 {
            uri: "s3://bucket/data1".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let source2 = SourceDescriptorV0 {
            uri: "s3://bucket/data2".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw1",
                TrustClass::Trusted,
                source1,
                None,
                &ingest,
            )
            .unwrap();
        session
            .register_source(
                "dataset://ns/raw2",
                TrustClass::Trusted,
                source2,
                None,
                &ingest,
            )
            .unwrap();

        // Transform with raw1 as input
        let transform = make_transform_node(
            "transform/clean",
            &["dataset://ns/raw1"],
            &["dataset://ns/clean_a"],
            ExecutionTrust::Core,
        );
        session
            .materialize_node_outputs(
                &transform,
                &[OutputSpec {
                    asset_key: "dataset://ns/clean_a".to_string(),
                    schema: None,
                    rows: Some(100),
                    bytes: Some(1000),
                }],
                1000,
                false,
                50,
            )
            .unwrap();
        let fp_a = session
            .fingerprint("dataset://ns/clean_a")
            .unwrap()
            .to_string();

        // Transform with raw2 as input (same transform node, different upstream)
        let transform2 = make_transform_node(
            "transform/clean",
            &["dataset://ns/raw2"],
            &["dataset://ns/clean_b"],
            ExecutionTrust::Core,
        );
        session
            .materialize_node_outputs(
                &transform2,
                &[OutputSpec {
                    asset_key: "dataset://ns/clean_b".to_string(),
                    schema: None,
                    rows: Some(100),
                    bytes: Some(1000),
                }],
                1001,
                false,
                50,
            )
            .unwrap();
        let fp_b = session
            .fingerprint("dataset://ns/clean_b")
            .unwrap()
            .to_string();

        assert_ne!(
            fp_a, fp_b,
            "fingerprint should change when upstream changes"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn multi_output_fingerprints_unique() {
        let base = temp_dir("multi_output_fp_unique");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([25u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register source
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        // Single node producing TWO outputs with SAME schema (None)
        let transform = make_transform_node(
            "transform/split",
            &["dataset://ns/raw"],
            &["dataset://ns/left", "dataset://ns/right"],
            ExecutionTrust::Core,
        );
        session
            .materialize_node_outputs(
                &transform,
                &[
                    OutputSpec {
                        asset_key: "dataset://ns/left".to_string(),
                        schema: None,
                        rows: Some(50),
                        bytes: Some(500),
                    },
                    OutputSpec {
                        asset_key: "dataset://ns/right".to_string(),
                        schema: None,
                        rows: Some(50),
                        bytes: Some(500),
                    },
                ],
                1000,
                false,
                50,
            )
            .unwrap();

        let fp_left = session
            .fingerprint("dataset://ns/left")
            .unwrap()
            .to_string();
        let fp_right = session
            .fingerprint("dataset://ns/right")
            .unwrap()
            .to_string();

        assert_ne!(
            fp_left, fp_right,
            "multi-output fingerprints should be unique even with same schema"
        );
        assert_eq!(fp_left.len(), 64, "fingerprint should be 64-char hex");
        assert_eq!(fp_right.len(), 64, "fingerprint should be 64-char hex");

        let _ = fs::remove_dir_all(&base);
    }
    #[test]
    fn lineage_dedupes_repeated_edges() {
        let base = temp_dir("lineage_dedupe");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([30u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register source
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        // Materialize same output twice
        let transform = make_transform_node(
            "transform/clean",
            &["dataset://ns/raw"],
            &["dataset://ns/clean"],
            ExecutionTrust::Core,
        );
        session
            .materialize_node_outputs(
                &transform,
                &[OutputSpec {
                    asset_key: "dataset://ns/clean".to_string(),
                    schema: None,
                    rows: Some(100),
                    bytes: Some(1000),
                }],
                1000,
                false,
                50,
            )
            .unwrap();
        session
            .materialize_node_outputs(
                &transform,
                &[OutputSpec {
                    asset_key: "dataset://ns/clean".to_string(),
                    schema: None,
                    rows: Some(100),
                    bytes: Some(1000),
                }],
                2000,
                true,
                10,
            )
            .unwrap();

        // Read lineage.json and check edge count
        let lineage_path = sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("lineage.json");
        let lineage: DatasetLineageV1 = read_json(&lineage_path).unwrap();

        // Should have exactly one edge (deduped)
        assert_eq!(
            lineage.edges.len(),
            1,
            "lineage should dedupe repeated edges"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn snapshot_determinism() {
        let base = temp_dir("snapshot_determinism");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([40u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register source and transform
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        let transform = make_transform_node(
            "transform/clean",
            &["dataset://ns/raw"],
            &["dataset://ns/clean"],
            ExecutionTrust::Core,
        );
        session
            .materialize_node_outputs(
                &transform,
                &[OutputSpec {
                    asset_key: "dataset://ns/clean".to_string(),
                    schema: None,
                    rows: Some(100),
                    bytes: Some(1000),
                }],
                1000,
                false,
                50,
            )
            .unwrap();

        // Read registry.json contents
        let registry_path = sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("registry.json");
        let content1 = fs::read_to_string(&registry_path).unwrap();

        // Flush again (no state change)
        session.finalize().unwrap();
        let content2 = fs::read_to_string(&registry_path).unwrap();

        assert_eq!(
            content1, content2,
            "registry.json should be byte-identical on repeated flush"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn trust_propagates_from_untrusted_input() {
        let base = temp_dir("trust_untrusted_input");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([50u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register UNTRUSTED source
        let source = SourceDescriptorV0 {
            uri: "http://external/data".to_string(),
            content_type: "application/json".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/external");
        session
            .register_source(
                "dataset://ns/external",
                TrustClass::Untrusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        // Core node reading untrusted input -> output should be Untrusted
        let transform = make_transform_node(
            "transform/process",
            &["dataset://ns/external"],
            &["dataset://ns/processed"],
            ExecutionTrust::Core,
        );
        session
            .materialize_node_outputs(
                &transform,
                &[OutputSpec {
                    asset_key: "dataset://ns/processed".to_string(),
                    schema: None,
                    rows: Some(10),
                    bytes: Some(100),
                }],
                1000,
                false,
                10,
            )
            .unwrap();

        // Read registry and check output trust
        let registry_path = sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("registry.json");
        let registry: DatasetRegistryV1 = read_json(&registry_path).unwrap();

        let output = registry
            .datasets
            .iter()
            .find(|d| d.asset_key == "dataset://ns/processed")
            .unwrap();
        assert!(
            matches!(output.trust, TrustClass::Untrusted),
            "output should be Untrusted when input is Untrusted"
        );

        // Check materialization unsafe_surface
        let mat_path = sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("materializations.ndjson");
        let content = fs::read_to_string(&mat_path).unwrap();
        assert!(
            content.contains("\"unsafe_surface\":true"),
            "materialization should have unsafe_surface=true"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn trust_propagates_from_unsafe_extension() {
        let base = temp_dir("trust_unsafe_extension");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([60u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register TRUSTED source
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/s3");
        session
            .register_source(
                "dataset://ns/trusted",
                TrustClass::Trusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        // UnsafeExtension node -> output should be Untrusted
        let transform = make_transform_node(
            "transform/custom",
            &["dataset://ns/trusted"],
            &["dataset://ns/custom_out"],
            ExecutionTrust::UnsafeExtension,
        );
        session
            .materialize_node_outputs(
                &transform,
                &[OutputSpec {
                    asset_key: "dataset://ns/custom_out".to_string(),
                    schema: None,
                    rows: Some(10),
                    bytes: Some(100),
                }],
                1000,
                false,
                10,
            )
            .unwrap();

        // Read registry and check output trust
        let registry_path = sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("registry.json");
        let registry: DatasetRegistryV1 = read_json(&registry_path).unwrap();

        let output = registry
            .datasets
            .iter()
            .find(|d| d.asset_key == "dataset://ns/custom_out")
            .unwrap();
        assert!(
            matches!(output.trust, TrustClass::Untrusted),
            "output should be Untrusted when execution_trust is UnsafeExtension"
        );

        let _ = fs::remove_dir_all(&base);
    }

    // ── Phase 3A: Emitter Correctness Gate tests ────────────────────

    #[test]
    fn materialize_fails_on_missing_input_asset() {
        let base = temp_dir("missing_input");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([70u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Do NOT register "dataset://ns/raw" — it's missing from registry
        let transform = make_transform_node(
            "transform/clean",
            &["dataset://ns/raw"],
            &["dataset://ns/clean"],
            ExecutionTrust::Core,
        );
        let result = session.materialize_node_outputs(
            &transform,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(100),
                bytes: Some(1000),
            }],
            1000,
            false,
            50,
        );

        assert!(result.is_err(), "should fail on missing input asset");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("missing input asset"),
            "error should mention missing input: {err_msg}"
        );
        assert!(
            err_msg.contains("dataset://ns/raw"),
            "error should name the missing asset: {err_msg}"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn lineage_in_place_uses_pre_mutation_fingerprint() {
        // Verify that lineage edges reference the INPUT fingerprint from BEFORE
        // the output was inserted into the registry. This prevents corruption
        // when an output asset_key matches an input asset_key (re-materialization).
        let base = temp_dir("lineage_premutation");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([71u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register source
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        // Capture the source fingerprint BEFORE transform
        let source_fp = session.fingerprint("dataset://ns/raw").unwrap().to_string();

        // Transform: raw → clean
        let transform = make_transform_node(
            "transform/clean",
            &["dataset://ns/raw"],
            &["dataset://ns/clean"],
            ExecutionTrust::Core,
        );
        session
            .materialize_node_outputs(
                &transform,
                &[OutputSpec {
                    asset_key: "dataset://ns/clean".to_string(),
                    schema: None,
                    rows: Some(100),
                    bytes: Some(1000),
                }],
                1000,
                false,
                50,
            )
            .unwrap();

        // Read lineage and verify input fingerprint matches the pre-mutation source fp
        let lineage_path = sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("lineage.json");
        let lineage: DatasetLineageV1 = read_json(&lineage_path).unwrap();

        assert_eq!(lineage.edges.len(), 1, "should have exactly one edge");
        assert_eq!(
            lineage.edges[0].input_fingerprint_v0, source_fp,
            "lineage input fingerprint must reference pre-mutation state"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn materialize_rejects_output_not_declared_in_node() {
        let base = temp_dir("undeclared_output");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([72u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register source
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        // Node declares output "dataset://ns/clean" but we try to materialize "dataset://ns/WRONG"
        let transform = make_transform_node(
            "transform/clean",
            &["dataset://ns/raw"],
            &["dataset://ns/clean"],
            ExecutionTrust::Core,
        );
        let result = session.materialize_node_outputs(
            &transform,
            &[OutputSpec {
                asset_key: "dataset://ns/WRONG".to_string(),
                schema: None,
                rows: Some(10),
                bytes: Some(100),
            }],
            1000,
            false,
            50,
        );

        assert!(result.is_err(), "should reject undeclared output");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not declared"),
            "error should mention not declared: {err_msg}"
        );
        assert!(
            err_msg.contains("dataset://ns/WRONG"),
            "error should name the undeclared output: {err_msg}"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn materialize_rejects_duplicate_output_asset_keys() {
        let base = temp_dir("duplicate_output");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([73u8; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        let mut session = DataOpsSession::new(Arc::clone(&sink));

        // Register source
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/data".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        };
        let ingest = make_source_node("ingest/v1");
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                source,
                None,
                &ingest,
            )
            .unwrap();

        // Node declares output A, but OutputSpec has A twice (duplicate)
        let transform = make_transform_node(
            "transform/dup",
            &["dataset://ns/raw"],
            &["dataset://ns/out"],
            ExecutionTrust::Core,
        );
        let result = session.materialize_node_outputs(
            &transform,
            &[
                OutputSpec {
                    asset_key: "dataset://ns/out".to_string(),
                    schema: None,
                    rows: Some(10),
                    bytes: Some(100),
                },
                OutputSpec {
                    asset_key: "dataset://ns/out".to_string(),
                    schema: None,
                    rows: Some(20),
                    bytes: Some(200),
                },
            ],
            1000,
            false,
            50,
        );

        assert!(result.is_err(), "should reject duplicate output keys");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("duplicate"),
            "error should mention duplicate: {err_msg}"
        );

        let _ = fs::remove_dir_all(&base);
    }
}
