use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

use sha2::{Digest, Sha256};
use swarm_torch_core::dataops::{
    validate_source_descriptor_bounds, DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1,
    LineageEdgeV1, MaterializationRecordCompat, MaterializationRecordV2,
};
use swarm_torch_core::observe::{EventRecord, MetricRecord, SpanRecord};
use swarm_torch_core::run_graph::GraphV1;

use crate::artifacts::RunArtifactBundle;

use super::model::Report;

pub fn load_report(run_dir: impl AsRef<Path>) -> io::Result<Report> {
    let (report, _) = load_report_with_warnings(run_dir)?;
    Ok(report)
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LoadWarning {
    SourceDescriptorBoundsExceeded { asset_key: String, message: String },
    SnapshotPairMismatch { message: String },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SnapshotPairCommitV1 {
    schema_version: u32,
    pair_seq: u64,
    registry_sha256: String,
    lineage_sha256: String,
}

const SNAPSHOT_PAIR_SCHEMA_V1: u32 = 1;

pub fn load_report_with_warnings(
    run_dir: impl AsRef<Path>,
) -> io::Result<(Report, Vec<LoadWarning>)> {
    let run_dir = run_dir.as_ref().to_path_buf();
    let bundle = RunArtifactBundle::open(&run_dir)?;
    let mut warnings = Vec::new();

    // Enforce tamper-evidence by default.
    bundle.validate_manifest()?;

    let mut graph: GraphV1 = read_json(run_dir.join("graph.json"))?;
    graph = graph
        .normalize()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let registry_updates: Vec<DatasetEntryV1> =
        read_ndjson_if_exists(run_dir.join("datasets").join("registry_updates.ndjson"))?;
    let lineage_updates: Vec<LineageEdgeV1> =
        read_ndjson_if_exists(run_dir.join("datasets").join("lineage_edges.ndjson"))?;
    let datasets_dir = run_dir.join("datasets");

    let pair_mismatch = snapshot_pair_mismatch_reason(&datasets_dir)?;
    let (registry_snapshot, lineage_snapshot) = if let Some(message) = pair_mismatch {
        warnings.push(LoadWarning::SnapshotPairMismatch { message });
        (DatasetRegistryV1::default(), DatasetLineageV1::default())
    } else {
        (
            read_json(datasets_dir.join("registry.json"))?,
            read_json(datasets_dir.join("lineage.json"))?,
        )
    };

    let registry = apply_registry_updates(registry_snapshot, registry_updates);
    let lineage = apply_lineage_updates(lineage_snapshot, lineage_updates);

    // M-12: Read-path descriptor bounds check (warn-and-continue) with surfaced warnings.
    for entry in &registry.datasets {
        if let Some(ref source) = entry.source {
            if let Err(e) = validate_source_descriptor_bounds(source) {
                warnings.push(LoadWarning::SourceDescriptorBoundsExceeded {
                    asset_key: entry.asset_key.clone(),
                    message: e.to_string(),
                });
            }
        }
    }

    let spans: Vec<SpanRecord> = read_ndjson(run_dir.join("spans.ndjson"))?;
    let events: Vec<EventRecord> = read_ndjson(run_dir.join("events.ndjson"))?;
    let metrics: Vec<MetricRecord> = read_ndjson(run_dir.join("metrics.ndjson"))?;
    let materializations_raw: Vec<MaterializationRecordCompat> =
        read_ndjson(run_dir.join("datasets").join("materializations.ndjson"))?;
    let mut materializations: Vec<MaterializationRecordV2> = materializations_raw
        .into_iter()
        .enumerate()
        .map(|(idx, record)| {
            let mut normalized = record.into_v2();
            if normalized.record_seq == 0 {
                normalized.record_seq = (idx as u64) + 1;
            }
            normalized
        })
        .collect();
    materializations.sort_by_key(|m| (m.ts_unix_nanos, m.record_seq));

    Ok((
        Report {
            run_dir,
            graph,
            registry,
            lineage,
            materializations,
            spans,
            events,
            metrics,
        },
        warnings,
    ))
}

fn snapshot_pair_mismatch_reason(datasets_dir: &Path) -> io::Result<Option<String>> {
    let marker_path = datasets_dir.join("snapshot_pair_commit.json");
    if !marker_path.exists() {
        return Ok(None);
    }

    let marker: SnapshotPairCommitV1 = read_json(&marker_path)?;
    if marker.schema_version != SNAPSHOT_PAIR_SCHEMA_V1 {
        return Ok(Some(format!(
            "unsupported snapshot_pair_commit schema_version: {}",
            marker.schema_version
        )));
    }

    let registry_path = datasets_dir.join("registry.json");
    let lineage_path = datasets_dir.join("lineage.json");
    let actual_registry = hash_file_sha256_hex(&registry_path)?;
    let actual_lineage = hash_file_sha256_hex(&lineage_path)?;

    if marker.registry_sha256 != actual_registry || marker.lineage_sha256 != actual_lineage {
        return Ok(Some(
            "snapshot pair hash mismatch; replaying from NDJSON updates".to_string(),
        ));
    }
    Ok(None)
}

fn hash_file_sha256_hex(path: &Path) -> io::Result<String> {
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(&bytes);
    Ok(hex_lower(&digest))
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

fn read_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<T> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_ndjson<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<Vec<T>> {
    let f = std::fs::File::open(path)?;
    let reader = io::BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v = serde_json::from_str::<T>(&line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid ndjson at line {}: {}", i + 1, e),
            )
        })?;
        out.push(v);
    }
    Ok(out)
}

fn read_ndjson_if_exists<T: serde::de::DeserializeOwned>(
    path: impl AsRef<Path>,
) -> io::Result<Vec<T>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }
    read_ndjson(path)
}

fn apply_registry_updates(
    snapshot: DatasetRegistryV1,
    updates: Vec<DatasetEntryV1>,
) -> DatasetRegistryV1 {
    let mut datasets: BTreeMap<String, DatasetEntryV1> = BTreeMap::new();
    for entry in snapshot.datasets {
        datasets.insert(entry.asset_key.clone(), entry);
    }
    for entry in updates {
        datasets.insert(entry.asset_key.clone(), entry);
    }
    DatasetRegistryV1 {
        schema_version: snapshot.schema_version,
        datasets: datasets.into_values().collect(),
    }
}

fn apply_lineage_updates(
    snapshot: DatasetLineageV1,
    updates: Vec<LineageEdgeV1>,
) -> DatasetLineageV1 {
    let mut edges: BTreeMap<(String, String, String), LineageEdgeV1> = BTreeMap::new();
    for edge in snapshot.edges {
        let key = (
            edge.input_fingerprint_v0.clone(),
            edge.output_fingerprint_v0.clone(),
            edge.node_id.to_string(),
        );
        edges.insert(key, edge);
    }
    for edge in updates {
        let key = (
            edge.input_fingerprint_v0.clone(),
            edge.output_fingerprint_v0.clone(),
            edge.node_id.to_string(),
        );
        edges.insert(key, edge);
    }
    DatasetLineageV1 {
        schema_version: snapshot.schema_version,
        edges: edges.into_values().collect(),
    }
}
