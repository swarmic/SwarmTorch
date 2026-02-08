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
        node_def_hash_v1, node_id_from_key, AssetRefV1, CanonParams, ExecutionTrust, GraphV1,
        NodeV1, OpKind,
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
}
