use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use swarm_torch_core::dataops::{
    DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1, LineageEdgeV1, MaterializationRecordV1,
    MaterializationRecordV2,
};
use swarm_torch_core::observe::{
    validate_event_record, validate_metric_record, validate_span_record, EventRecord, MetricRecord,
    RunId, SpanRecord,
};
use swarm_torch_core::run_graph::{validate_graph_v1, validate_node_v1, GraphV1};

use super::io::{
    append_ndjson, collect_files_recursive, ensure_file, hex_lower, read_json, rel_path_string,
    sha256_file, write_json_pretty_atomic,
};
use super::record_validation_error_to_io;

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
    /// This validates node field bounds (M-09) and computes derived fields
    /// (`node_id`, `node_def_hash`) according to ADR-0017.
    ///
    /// Returns `Err(InvalidData)` if any node violates M-09 field bounds.
    pub fn write_graph(&self, graph: &GraphV1) -> io::Result<()> {
        let mut g = graph.clone();
        // M-09: validate node field bounds before persist (fail-closed).
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
        validate_span_record(span).map_err(record_validation_error_to_io)?;
        append_ndjson(&self.run_dir.join("spans.ndjson"), span)
    }

    pub fn append_event(&self, event: &EventRecord) -> io::Result<()> {
        validate_event_record(event).map_err(record_validation_error_to_io)?;
        append_ndjson(&self.run_dir.join("events.ndjson"), event)
    }

    pub fn append_metric(&self, metric: &MetricRecord) -> io::Result<()> {
        validate_metric_record(metric).map_err(record_validation_error_to_io)?;
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

    pub fn append_materialization_v2(
        &self,
        materialization: &MaterializationRecordV2,
    ) -> io::Result<()> {
        append_ndjson(
            &self
                .run_dir
                .join("datasets")
                .join("materializations.ndjson"),
            materialization,
        )
    }

    pub fn append_registry_update(&self, dataset: &DatasetEntryV1) -> io::Result<()> {
        append_ndjson(
            &self
                .run_dir
                .join("datasets")
                .join("registry_updates.ndjson"),
            dataset,
        )
    }

    pub fn append_lineage_edge_update(&self, edge: &LineageEdgeV1) -> io::Result<()> {
        append_ndjson(
            &self.run_dir.join("datasets").join("lineage_edges.ndjson"),
            edge,
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
        let canonical_root = self.run_dir.canonicalize()?;

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
        let canonical_root = self.run_dir.canonicalize()?;

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

        let mut seen_paths: BTreeSet<String> = BTreeSet::new();
        let mut seen_required_paths: BTreeSet<String> = BTreeSet::new();

        for entry in manifest.entries {
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

            let path = self.run_dir.join(&entry.path);
            if !path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("missing file listed in manifest: {}", entry.path),
                ));
            }
            ensure_path_within_bundle(&path, &canonical_root, &entry.path)?;
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

        for required in required_paths_v1() {
            if !seen_required_paths.contains(*required) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("required manifest entry missing: {required}"),
                ));
            }
        }

        Ok(())
    }
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
        "datasets/registry_updates.ndjson",
        "datasets/lineage_edges.ndjson",
    ]
}

fn is_required_path_v1(p: &str) -> bool {
    required_paths_v1().contains(&p)
}

fn validate_manifest_path(path: &str) -> io::Result<()> {
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
