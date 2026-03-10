use super::*;
use crate::artifacts::{
    ArtifactWriteProfile, DataOpsSession, ManifestRefreshPolicy, RunArtifactBundle,
    RunArtifactSink, SnapshotProfile,
};
use crate::report::render::{render_html, render_timeline};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use swarm_torch_core::dataops::{
    DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1, MaterializationRecordV1,
    MaterializationRecordV2, MaterializationStatusV0, SourceDescriptorV0, TransformAuditV0,
    TrustClass, UnsafeReasonV0, MATERIALIZATION_SCHEMA_V2, MAX_SOURCE_URI_LEN,
};
use swarm_torch_core::observe::{RunId, TraceId};
use swarm_torch_core::run_graph::{
    AssetRefV1, CanonParams, ExecutionTrust, GraphV1, NodeV1, OpKind,
};

fn temp_dir(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut dir = std::env::temp_dir();
    dir.push(format!("swarmtorch_report_{prefix}_{pid}_{nanos}"));
    dir
}

fn make_node(key: &str, trust: ExecutionTrust, inputs: &[&str]) -> NodeV1 {
    NodeV1 {
        node_key: key.to_string(),
        node_id: None,
        op_kind: OpKind::Data,
        op_type: "test".to_string(),
        inputs: inputs
            .iter()
            .map(|k| AssetRefV1 {
                asset_key: k.to_string(),
                fingerprint: None,
            })
            .collect(),
        outputs: vec![],
        params: CanonParams::new(),
        code_ref: Some("test@0.1.0".to_string()),
        unsafe_surface: false,
        execution_trust: trust,
        node_def_hash: None,
        execution_hint: None,
        cache_policy: None,
        materialization_policy: None,
        resources: None,
        op_hash: None,
    }
}

fn make_entry(asset_key: &str, trust: TrustClass) -> DatasetEntryV1 {
    DatasetEntryV1 {
        asset_key: asset_key.to_string(),
        fingerprint_v0: "a".repeat(64),
        source_fingerprint_v0: "b".repeat(64),
        schema_hash_v0: "c".repeat(64),
        recipe_hash_v0: "d".repeat(64),
        trust,
        source: None,
        schema: None,
        license_flags: vec![],
        pii_tags: vec![],
    }
}

fn legacy_is_node_unsafe(node: &NodeV1, registry: &DatasetRegistryV1) -> bool {
    if node.execution_trust != ExecutionTrust::Core {
        return true;
    }
    for input in &node.inputs {
        match registry
            .datasets
            .iter()
            .find(|entry| entry.asset_key == input.asset_key)
        {
            Some(entry) if entry.trust == TrustClass::Untrusted => return true,
            None => return true,
            _ => {}
        }
    }
    false
}

#[test]
fn is_node_unsafe_registry_index_matches_legacy_logic() {
    let node = make_node(
        "transform/index_equivalence",
        ExecutionTrust::Core,
        &["dataset://ns/raw", "dataset://ns/aux"],
    );
    let trusted_registry = DatasetRegistryV1 {
        schema_version: 1,
        datasets: vec![
            make_entry("dataset://ns/raw", TrustClass::Trusted),
            make_entry("dataset://ns/aux", TrustClass::Trusted),
        ],
    };
    let untrusted_registry = DatasetRegistryV1 {
        schema_version: 1,
        datasets: vec![
            make_entry("dataset://ns/raw", TrustClass::Trusted),
            make_entry("dataset://ns/aux", TrustClass::Untrusted),
        ],
    };
    let missing_registry = DatasetRegistryV1 {
        schema_version: 1,
        datasets: vec![make_entry("dataset://ns/raw", TrustClass::Trusted)],
    };

    assert_eq!(
        is_node_unsafe(&node, &trusted_registry),
        legacy_is_node_unsafe(&node, &trusted_registry)
    );
    assert_eq!(
        is_node_unsafe(&node, &untrusted_registry),
        legacy_is_node_unsafe(&node, &untrusted_registry)
    );
    assert_eq!(
        is_node_unsafe(&node, &missing_registry),
        legacy_is_node_unsafe(&node, &missing_registry)
    );
}

#[test]
fn report_marks_node_unsafe_on_untrusted_input() {
    let node = make_node(
        "transform/clean",
        ExecutionTrust::Core,
        &["dataset://ns/raw"],
    );
    let registry = DatasetRegistryV1 {
        schema_version: 1,
        datasets: vec![make_entry("dataset://ns/raw", TrustClass::Untrusted)],
    };

    assert!(
        is_node_unsafe(&node, &registry),
        "node with untrusted input should be marked unsafe"
    );
}

#[test]
fn report_marks_node_unsafe_on_missing_input_registry_entry() {
    // Node references an input that doesn't exist in the registry at all
    let node = make_node(
        "transform/clean",
        ExecutionTrust::Core,
        &["dataset://ns/missing"],
    );
    let registry = DatasetRegistryV1 {
        schema_version: 1,
        datasets: vec![], // empty registry
    };

    assert!(
        is_node_unsafe(&node, &registry),
        "node with missing input in registry should be marked unsafe (fail closed)"
    );
}

#[test]
fn report_node_safe_with_core_trust_and_trusted_inputs() {
    let node = make_node(
        "transform/clean",
        ExecutionTrust::Core,
        &["dataset://ns/raw"],
    );
    let registry = DatasetRegistryV1 {
        schema_version: 1,
        datasets: vec![make_entry("dataset://ns/raw", TrustClass::Trusted)],
    };

    assert!(
        !is_node_unsafe(&node, &registry),
        "Core trust node with all trusted inputs should not be unsafe"
    );
}

#[test]
fn timeline_materialization_includes_node_id_and_node_def_hash() {
    let node_id = TraceId::from_bytes([1u8; 16]);
    let mut node = make_node(
        "transform/clean",
        ExecutionTrust::Core,
        &["dataset://ns/missing"],
    );
    node.node_id = Some(node_id);

    let mat = MaterializationRecordV2 {
        schema_version: MATERIALIZATION_SCHEMA_V2,
        record_seq: 1,
        ts_unix_nanos: 1000,
        asset_key: "dataset://ns/out".to_string(),
        fingerprint_v0: "x".repeat(64),
        node_id,
        node_def_hash: "h".repeat(64),
        op_type: "transform".to_string(),
        input_asset_keys: vec!["dataset://ns/missing".to_string()],
        input_fingerprints_v0: vec!["y".repeat(64)],
        rows: Some(100),
        bytes: Some(1000),
        cache_decision: swarm_torch_core::dataops::CacheDecisionV0::Miss,
        cache_reason: None,
        cache_key_v0: None,
        cache_hit: Some(false),
        duration_ms: Some(50),
        unsafe_surface: false, // intentionally false: timeline should derive from node+registry.
        unsafe_reasons: Vec::new(),
        applied_transforms: Vec::new(),
        status: MaterializationStatusV0::Ok,
        error_code: None,
        quality: None,
    };

    let report = Report {
        run_dir: PathBuf::from("/tmp/test"),
        graph: GraphV1 {
            schema_version: 1,
            graph_id: None,
            nodes: vec![node],
            edges: vec![],
        },
        registry: DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![], // missing input => derived unsafe=true
        },
        lineage: DatasetLineageV1 {
            schema_version: 1,
            edges: vec![],
        },
        materializations: vec![mat],
        spans: vec![],
        events: vec![],
        metrics: vec![],
    };
    let timeline_html = render_timeline(&report);

    assert!(
        timeline_html.contains("node_id="),
        "timeline detail should include node_id: {timeline_html}"
    );
    assert!(
        timeline_html.contains("node_def_hash="),
        "timeline detail should include node_def_hash: {timeline_html}"
    );
    assert!(
        timeline_html.contains(&"h".repeat(64)),
        "timeline detail should include full node_def_hash value: {timeline_html}"
    );
    assert!(
            timeline_html.contains("unsafe=true"),
            "timeline unsafe should be derived from node+registry, not materialization flag: {timeline_html}"
        );
    assert!(
        timeline_html.contains("unsafe_reasons=none"),
        "timeline detail should include unsafe_reasons for explainability: {timeline_html}"
    );
}

#[test]
fn timeline_materialization_includes_unsafe_reasons_when_present() {
    let node_id = TraceId::from_bytes([9u8; 16]);
    let mat = MaterializationRecordV2 {
        schema_version: MATERIALIZATION_SCHEMA_V2,
        record_seq: 1,
        ts_unix_nanos: 1234,
        asset_key: "dataset://ns/out".to_string(),
        fingerprint_v0: "x".repeat(64),
        node_id,
        node_def_hash: "h".repeat(64),
        op_type: "transform".to_string(),
        input_asset_keys: vec!["dataset://ns/in".to_string()],
        input_fingerprints_v0: vec!["y".repeat(64)],
        rows: Some(1),
        bytes: Some(2),
        cache_decision: swarm_torch_core::dataops::CacheDecisionV0::Miss,
        cache_reason: None,
        cache_key_v0: None,
        cache_hit: Some(false),
        duration_ms: Some(3),
        unsafe_surface: true,
        unsafe_reasons: vec![
            UnsafeReasonV0::UntrustedInput,
            UnsafeReasonV0::UnsafeExtension,
        ],
        applied_transforms: Vec::new(),
        status: MaterializationStatusV0::Ok,
        error_code: None,
        quality: None,
    };

    let report = Report {
        run_dir: PathBuf::from("/tmp/test"),
        graph: GraphV1 {
            schema_version: 1,
            graph_id: None,
            nodes: vec![],
            edges: vec![],
        },
        registry: DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![],
        },
        lineage: DatasetLineageV1 {
            schema_version: 1,
            edges: vec![],
        },
        materializations: vec![mat],
        spans: vec![],
        events: vec![],
        metrics: vec![],
    };

    let timeline_html = render_timeline(&report);
    assert!(
        timeline_html.contains("unsafe_reasons=untrusted_input,unsafe_extension"),
        "timeline should render serialized unsafe reason labels: {timeline_html}"
    );
}

#[test]
fn report_warning_lists_unsafe_materialization_reasons() {
    let node_id = TraceId::from_bytes([10u8; 16]);
    let mat = MaterializationRecordV2 {
        schema_version: MATERIALIZATION_SCHEMA_V2,
        record_seq: 1,
        ts_unix_nanos: 1234,
        asset_key: "dataset://ns/out".to_string(),
        fingerprint_v0: "x".repeat(64),
        node_id,
        node_def_hash: "h".repeat(64),
        op_type: "transform".to_string(),
        input_asset_keys: vec!["dataset://ns/in".to_string()],
        input_fingerprints_v0: vec!["y".repeat(64)],
        rows: Some(1),
        bytes: Some(2),
        cache_decision: swarm_torch_core::dataops::CacheDecisionV0::Miss,
        cache_reason: None,
        cache_key_v0: None,
        cache_hit: Some(false),
        duration_ms: Some(3),
        unsafe_surface: true,
        unsafe_reasons: vec![UnsafeReasonV0::MissingProvenance],
        applied_transforms: Vec::new(),
        status: MaterializationStatusV0::Ok,
        error_code: None,
        quality: None,
    };

    let report = Report {
        run_dir: PathBuf::from("/tmp/test"),
        graph: GraphV1 {
            schema_version: 1,
            graph_id: None,
            nodes: vec![],
            edges: vec![],
        },
        registry: DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![],
        },
        lineage: DatasetLineageV1 {
            schema_version: 1,
            edges: vec![],
        },
        materializations: vec![mat],
        spans: vec![],
        events: vec![],
        metrics: vec![],
    };

    let html = render_html(&report);
    assert!(
        html.contains("unsafe materialization"),
        "report warning banner should list unsafe materializations: {html}"
    );
    assert!(
        html.contains("reasons="),
        "report warning banner should show reason labels: {html}"
    );
    assert!(
        html.contains("missing_provenance"),
        "report warning banner should render missing_provenance label: {html}"
    );
}

#[test]
fn report_loads_materialization_v1_and_v2() {
    let base = temp_dir("compat_v1_v2");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([77u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let node_id = TraceId::from_bytes([2u8; 16]);

    let v1 = MaterializationRecordV1 {
        schema_version: 1,
        ts_unix_nanos: 1000,
        asset_key: "dataset://ns/v1".to_string(),
        fingerprint_v0: "a".repeat(64),
        node_id,
        node_def_hash: "b".repeat(64),
        rows: Some(1),
        bytes: Some(2),
        cache_hit: Some(false),
        duration_ms: Some(3),
        quality_flags: None,
        unsafe_surface: false,
    };
    bundle.append_materialization(&v1).unwrap();

    let v2 = MaterializationRecordV2 {
        schema_version: MATERIALIZATION_SCHEMA_V2,
        record_seq: 10,
        ts_unix_nanos: 2000,
        asset_key: "dataset://ns/v2".to_string(),
        fingerprint_v0: "c".repeat(64),
        node_id,
        node_def_hash: "d".repeat(64),
        op_type: "transform".to_string(),
        input_asset_keys: vec!["dataset://ns/in".to_string()],
        input_fingerprints_v0: vec!["e".repeat(64)],
        rows: Some(4),
        bytes: Some(5),
        duration_ms: Some(6),
        cache_decision: swarm_torch_core::dataops::CacheDecisionV0::Hit,
        cache_reason: Some("test".to_string()),
        cache_key_v0: Some("f".repeat(64)),
        cache_hit: Some(true),
        unsafe_surface: true,
        unsafe_reasons: Vec::new(),
        applied_transforms: Vec::new(),
        status: MaterializationStatusV0::Ok,
        error_code: None,
        quality: None,
    };
    bundle.append_materialization_v2(&v2).unwrap();
    bundle.finalize_manifest().unwrap();

    let report = load_report(bundle.run_dir()).unwrap();
    assert_eq!(report.materializations.len(), 2);
    assert_eq!(
        report.materializations[0].schema_version,
        MATERIALIZATION_SCHEMA_V2
    );
    assert_eq!(
        report.materializations[1].schema_version,
        MATERIALIZATION_SCHEMA_V2
    );
    assert_eq!(report.materializations[0].asset_key, "dataset://ns/v1");
    assert_eq!(report.materializations[1].asset_key, "dataset://ns/v2");
    assert!(
        report.materializations[0].applied_transforms.is_empty(),
        "v1 compatibility rows should default to empty applied_transforms"
    );
    assert!(
        report.materializations[1].applied_transforms.is_empty(),
        "v2 row with no transform audits should remain empty"
    );
}

#[test]
fn report_loads_record_with_and_without_applied_transforms() {
    let base = temp_dir("compat_applied_transforms");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([79u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let node_id = TraceId::from_bytes([3u8; 16]);

    let no_transforms = MaterializationRecordV2 {
        schema_version: MATERIALIZATION_SCHEMA_V2,
        record_seq: 1,
        ts_unix_nanos: 1000,
        asset_key: "dataset://ns/no_transform".to_string(),
        fingerprint_v0: "a".repeat(64),
        node_id,
        node_def_hash: "b".repeat(64),
        op_type: "transform".to_string(),
        input_asset_keys: vec![],
        input_fingerprints_v0: vec![],
        rows: None,
        bytes: None,
        duration_ms: None,
        cache_decision: swarm_torch_core::dataops::CacheDecisionV0::Unknown,
        cache_reason: None,
        cache_key_v0: None,
        cache_hit: None,
        unsafe_surface: false,
        unsafe_reasons: Vec::new(),
        applied_transforms: Vec::new(),
        status: MaterializationStatusV0::Ok,
        error_code: None,
        quality: None,
    };
    bundle.append_materialization_v2(&no_transforms).unwrap();

    let with_transforms = MaterializationRecordV2 {
        schema_version: MATERIALIZATION_SCHEMA_V2,
        record_seq: 2,
        ts_unix_nanos: 2000,
        asset_key: "dataset://ns/with_transform".to_string(),
        fingerprint_v0: "c".repeat(64),
        node_id,
        node_def_hash: "d".repeat(64),
        op_type: "transform".to_string(),
        input_asset_keys: vec![],
        input_fingerprints_v0: vec![],
        rows: None,
        bytes: None,
        duration_ms: None,
        cache_decision: swarm_torch_core::dataops::CacheDecisionV0::Unknown,
        cache_reason: None,
        cache_key_v0: None,
        cache_hit: None,
        unsafe_surface: true,
        unsafe_reasons: vec![UnsafeReasonV0::UnsafeExtension],
        applied_transforms: vec![TransformAuditV0 {
            transform_name: "dp_clip".to_string(),
            core_trusted: false,
            round_id: 11,
        }],
        status: MaterializationStatusV0::Ok,
        error_code: None,
        quality: None,
    };
    bundle.append_materialization_v2(&with_transforms).unwrap();
    bundle.finalize_manifest().unwrap();

    let report = load_report(bundle.run_dir()).unwrap();
    assert_eq!(report.materializations.len(), 2);
    assert!(report.materializations[0].applied_transforms.is_empty());
    assert_eq!(report.materializations[1].applied_transforms.len(), 1);
    assert_eq!(
        report.materializations[1].applied_transforms[0].transform_name,
        "dp_clip"
    );
}

#[test]
fn mid_run_report_succeeds_with_manifest_always_policy() {
    let base = temp_dir("mid_run_report_manifest_always");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([88u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let profile = ArtifactWriteProfile {
        snapshot_profile: SnapshotProfile::streaming(100),
        manifest_policy: ManifestRefreshPolicy::Always,
    };
    let sink = std::sync::Arc::new(RunArtifactSink::with_profile(bundle, profile));
    let mut session = DataOpsSession::with_profile(
        std::sync::Arc::clone(&sink),
        SnapshotProfile::streaming(100),
    );

    let ingest_node = NodeV1 {
        node_key: "ingest/raw".to_string(),
        node_id: None,
        op_kind: OpKind::Data,
        op_type: "ingest".to_string(),
        inputs: vec![],
        outputs: vec![AssetRefV1 {
            asset_key: "dataset://ns/raw".to_string(),
            fingerprint: None,
        }],
        params: CanonParams::new(),
        code_ref: Some("test@0.1.0".to_string()),
        unsafe_surface: false,
        execution_trust: ExecutionTrust::Core,
        node_def_hash: None,
        execution_hint: None,
        cache_policy: None,
        materialization_policy: None,
        resources: None,
        op_hash: None,
    };

    let source = SourceDescriptorV0 {
        uri: "s3://bucket/raw.parquet".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: Some("v1".to_string()),
    };
    session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            source,
            None,
            &ingest_node,
        )
        .unwrap();

    // No finalize() call here: this is intentionally a mid-run load.
    let report = load_report(sink.bundle().run_dir()).unwrap();
    assert!(
        report
            .registry
            .datasets
            .iter()
            .any(|dataset| dataset.asset_key == "dataset://ns/raw"),
        "report should replay registry updates for mid-run visibility"
    );
}

#[test]
fn load_report_backward_compatible_after_artifacts_module_split() {
    let base = temp_dir("compat_after_module_split");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([89u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let node_id = TraceId::from_bytes([7u8; 16]);
    let v1 = MaterializationRecordV1 {
        schema_version: 1,
        ts_unix_nanos: 1000,
        asset_key: "dataset://ns/v1".to_string(),
        fingerprint_v0: "a".repeat(64),
        node_id,
        node_def_hash: "b".repeat(64),
        rows: Some(10),
        bytes: Some(100),
        cache_hit: Some(false),
        duration_ms: Some(5),
        quality_flags: None,
        unsafe_surface: false,
    };
    bundle.append_materialization(&v1).unwrap();
    bundle.finalize_manifest().unwrap();

    let report = load_report(bundle.run_dir()).unwrap();
    assert_eq!(report.materializations.len(), 1);
    assert_eq!(
        report.materializations[0].schema_version, MATERIALIZATION_SCHEMA_V2,
        "legacy v1 rows should normalize to v2 on load"
    );
    assert!(
        report.materializations[0].applied_transforms.is_empty(),
        "missing transform metadata in legacy rows should default empty"
    );
}

#[test]
fn load_report_with_warnings_emits_descriptor_bounds_violations() {
    let base = temp_dir("load_report_with_warnings_descriptor_violation");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([90u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let oversized_entry = DatasetEntryV1 {
        asset_key: "dataset://ns/oversized".to_string(),
        fingerprint_v0: "a".repeat(64),
        source_fingerprint_v0: "b".repeat(64),
        schema_hash_v0: "c".repeat(64),
        recipe_hash_v0: "d".repeat(64),
        trust: TrustClass::Trusted,
        source: Some(SourceDescriptorV0 {
            uri: "x".repeat(MAX_SOURCE_URI_LEN + 1),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: None,
        }),
        schema: None,
        license_flags: vec![],
        pii_tags: vec![],
    };
    bundle
        .write_dataset_registry(&DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![oversized_entry],
        })
        .unwrap();
    bundle.finalize_manifest().unwrap();

    let (_report, warnings) = load_report_with_warnings(bundle.run_dir()).unwrap();
    assert!(
        warnings.iter().any(|warning| matches!(
            warning,
            LoadWarning::SourceDescriptorBoundsExceeded { asset_key, .. }
                if asset_key == "dataset://ns/oversized"
        )),
        "expected descriptor bounds warning, got: {warnings:?}"
    );
}

#[test]
fn load_report_with_warnings_prefers_replay_on_snapshot_pair_mismatch() {
    #[derive(serde::Serialize)]
    struct SnapshotPairCommitV1 {
        schema_version: u32,
        pair_seq: u64,
        registry_sha256: String,
        lineage_sha256: String,
    }

    let base = temp_dir("load_report_snapshot_pair_mismatch");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([91u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    // Snapshot has "old" dataset.
    bundle
        .write_dataset_registry(&DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![DatasetEntryV1 {
                asset_key: "dataset://ns/old".to_string(),
                fingerprint_v0: "1".repeat(64),
                source_fingerprint_v0: "2".repeat(64),
                schema_hash_v0: "3".repeat(64),
                recipe_hash_v0: "4".repeat(64),
                trust: TrustClass::Trusted,
                source: None,
                schema: None,
                license_flags: vec![],
                pii_tags: vec![],
            }],
        })
        .unwrap();

    // NDJSON update has "new" dataset.
    bundle
        .append_registry_update(&DatasetEntryV1 {
            asset_key: "dataset://ns/new".to_string(),
            fingerprint_v0: "a".repeat(64),
            source_fingerprint_v0: "b".repeat(64),
            schema_hash_v0: "c".repeat(64),
            recipe_hash_v0: "d".repeat(64),
            trust: TrustClass::Trusted,
            source: None,
            schema: None,
            license_flags: vec![],
            pii_tags: vec![],
        })
        .unwrap();

    // Write mismatched pair commit marker to force replay preference.
    let marker_path = bundle
        .run_dir()
        .join("datasets")
        .join("snapshot_pair_commit.json");
    std::fs::write(
        &marker_path,
        serde_json::to_vec_pretty(&SnapshotPairCommitV1 {
            schema_version: 1,
            pair_seq: 1,
            registry_sha256: "deadbeef".to_string(),
            lineage_sha256: "deadbeef".to_string(),
        })
        .unwrap(),
    )
    .unwrap();
    bundle.finalize_manifest().unwrap();

    let (report, warnings) = load_report_with_warnings(bundle.run_dir()).unwrap();
    assert!(
        warnings
            .iter()
            .any(|warning| matches!(warning, LoadWarning::SnapshotPairMismatch { .. })),
        "expected snapshot pair mismatch warning, got: {warnings:?}"
    );
    assert!(
        report
            .registry
            .datasets
            .iter()
            .any(|entry| entry.asset_key == "dataset://ns/new"),
        "replay should retain registry updates when snapshot pair is inconsistent"
    );
    assert!(
        !report
            .registry
            .datasets
            .iter()
            .any(|entry| entry.asset_key == "dataset://ns/old"),
        "inconsistent snapshot should be ignored in replay-preferred mode"
    );
}
