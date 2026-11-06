use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use sha2::{Digest, Sha256};
use swarm_torch_core::dataops::{
    validate_source_descriptor_bounds, DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1,
    LineageEdgeV1, MaterializationRecordCompat, MaterializationRecordV2,
};
use swarm_torch_core::observe::{EventRecord, MetricRecord, SpanRecord};
use swarm_torch_core::run_graph::GraphV1;

use crate::artifacts::{
    hex_lower, parse_json_slice, parse_ndjson_slice_bounded, RunArtifactBundle, TrustPolicy,
    VerifiedBundleV1, MAX_NDJSON_LINES, MAX_NDJSON_LINE_BYTES,
};

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

/// Load a report enforcing [`TrustPolicy::Strict`] (`ADR-0023` §3.1, `P2-10`, `W10-02`).
///
/// Fails closed if `policy` is not [`TrustPolicy::Strict`] or if the bundle's
/// [`crate::artifacts::ManifestSignatureV1`] is missing or unauthorized.
pub fn load_report_strict(
    run_dir: impl AsRef<Path>,
    policy: &TrustPolicy,
) -> io::Result<(Report, Vec<LoadWarning>)> {
    if !matches!(policy, TrustPolicy::Strict { .. }) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "load_report_strict requires TrustPolicy::Strict",
        ));
    }
    load_report_with_policy(run_dir, policy)
}

/// Load a report under the caller-specified [`TrustPolicy`], consuming only verified
/// manifest-listed bytes in a single pass with zero path reopens (`ADR-0023` §3.1).
pub fn load_report_with_policy(
    run_dir: impl AsRef<Path>,
    policy: &TrustPolicy,
) -> io::Result<(Report, Vec<LoadWarning>)> {
    let bundle = RunArtifactBundle::open(run_dir.as_ref())?;
    let verified = bundle.verify_and_load(policy)?;
    load_report_from_verified(&verified)
}

/// Parse a [`Report`] exclusively from an already-verified [`VerifiedBundleV1`] snapshot.
///
/// Never touches the filesystem or consults files absent from `verified.manifest().entries`.
pub fn load_report_from_verified(
    verified: &VerifiedBundleV1,
) -> io::Result<(Report, Vec<LoadWarning>)> {
    let mut warnings = Vec::new();

    let mut graph: GraphV1 = parse_json_slice(verified.require_bytes("graph.json")?)?;
    graph = graph
        .normalize()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    let registry_updates: Vec<DatasetEntryV1> = match verified
        .get_bytes("datasets/registry_updates.ndjson")
    {
        Some(bytes) => parse_ndjson_slice_bounded(bytes, MAX_NDJSON_LINE_BYTES, MAX_NDJSON_LINES)?,
        None => Vec::new(),
    };
    let lineage_updates: Vec<LineageEdgeV1> = match verified
        .get_bytes("datasets/lineage_edges.ndjson")
    {
        Some(bytes) => parse_ndjson_slice_bounded(bytes, MAX_NDJSON_LINE_BYTES, MAX_NDJSON_LINES)?,
        None => Vec::new(),
    };

    let pair_mismatch = snapshot_pair_mismatch_reason(verified)?;
    let (registry_snapshot, lineage_snapshot) = if let Some(message) = pair_mismatch {
        warnings.push(LoadWarning::SnapshotPairMismatch { message });
        (DatasetRegistryV1::default(), DatasetLineageV1::default())
    } else {
        (
            parse_json_slice(verified.require_bytes("datasets/registry.json")?)?,
            parse_json_slice(verified.require_bytes("datasets/lineage.json")?)?,
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

    let spans: Vec<SpanRecord> = parse_ndjson_slice_bounded(
        verified.require_bytes("spans.ndjson")?,
        MAX_NDJSON_LINE_BYTES,
        MAX_NDJSON_LINES,
    )?;
    let events: Vec<EventRecord> = parse_ndjson_slice_bounded(
        verified.require_bytes("events.ndjson")?,
        MAX_NDJSON_LINE_BYTES,
        MAX_NDJSON_LINES,
    )?;
    let metrics: Vec<MetricRecord> = parse_ndjson_slice_bounded(
        verified.require_bytes("metrics.ndjson")?,
        MAX_NDJSON_LINE_BYTES,
        MAX_NDJSON_LINES,
    )?;
    let materializations_raw: Vec<MaterializationRecordCompat> = parse_ndjson_slice_bounded(
        verified.require_bytes("datasets/materializations.ndjson")?,
        MAX_NDJSON_LINE_BYTES,
        MAX_NDJSON_LINES,
    )?;
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
            run_dir: verified.run_dir().to_path_buf(),
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

pub fn load_report_with_warnings(
    run_dir: impl AsRef<Path>,
) -> io::Result<(Report, Vec<LoadWarning>)> {
    load_report_with_policy(run_dir, &TrustPolicy::LegacyPermissive)
}

fn snapshot_pair_mismatch_reason(verified: &VerifiedBundleV1) -> io::Result<Option<String>> {
    // Only consult `datasets/snapshot_pair_commit.json` if it was explicitly listed and
    // digest-verified in `manifest.json` (`ADR-0023` §3.1).
    let Some(marker_bytes) = verified.get_bytes("datasets/snapshot_pair_commit.json") else {
        return Ok(None);
    };

    let marker: SnapshotPairCommitV1 = parse_json_slice(marker_bytes)?;
    if marker.schema_version != SNAPSHOT_PAIR_SCHEMA_V1 {
        return Ok(Some(format!(
            "unsupported snapshot_pair_commit schema_version: {}",
            marker.schema_version
        )));
    }

    let registry_bytes = verified.require_bytes("datasets/registry.json")?;
    let lineage_bytes = verified.require_bytes("datasets/lineage.json")?;
    let actual_registry = hex_lower(&Sha256::digest(registry_bytes));
    let actual_lineage = hex_lower(&Sha256::digest(lineage_bytes));

    if marker.registry_sha256 != actual_registry || marker.lineage_sha256 != actual_lineage {
        return Ok(Some(
            "snapshot pair hash mismatch; replaying from NDJSON updates".to_string(),
        ));
    }
    Ok(None)
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
