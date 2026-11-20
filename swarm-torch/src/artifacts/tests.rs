use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use swarm_torch_core::dataops::{
    cache_key_v0, CacheDecisionV0, DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1,
    LineageEdgeV1, MaterializationRecordCompat, MaterializationRecordV1, MaterializationStatusV0,
    OutputSpecCore, SourceDescriptorV0, TransformAuditV0, TrustClass, UnsafeReasonV0,
    MATERIALIZATION_SCHEMA_V2, MAX_ETAG_OR_VERSION_LEN, MAX_SOURCE_URI_LEN,
};
use swarm_torch_core::observe::{
    AttrMap, AttrValue, EventRecord, MetricRecord, RunEventEmitter, RunId, SpanId, SpanRecord,
    TraceId, MAX_METRIC_UNIT_LEN, MAX_RECORD_ATTRS, MAX_RECORD_NAME_LEN,
};
use swarm_torch_core::run_graph::{
    node_def_hash_v1, node_id_from_key, AssetRefV1, CanonParams, CanonValue, DeviceAffinity,
    ExecutionHint, ExecutionTrust, GraphV1, NodeV1, OpKind, PreferredProfile, MAX_NODE_KEY_LEN,
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
fn validate_manifest_rejects_missing_required_entries() {
    let base = temp_dir("manifest_missing_required");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([12u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    entries
        .retain(|entry| entry.get("path") != Some(&serde_json::Value::String("run.json".into())));
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("missing required manifest entry should fail");
    assert!(
        err.to_string().contains("required manifest entry missing"),
        "error should mention missing required entry: {err}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn validate_manifest_rejects_duplicate_paths() {
    let base = temp_dir("manifest_duplicate_paths");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([13u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    let duplicate = entries
        .first()
        .cloned()
        .expect("manifest must contain entries");
    entries.push(duplicate);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("duplicate path entries should fail");
    assert!(
        err.to_string().contains("duplicate manifest path entry"),
        "error should mention duplicate manifest path: {err}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn validate_manifest_rejects_parent_traversal_path() {
    let base = temp_dir("manifest_parent_traversal");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([14u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let outside = bundle
        .run_dir()
        .parent()
        .expect("run_dir parent should exist")
        .join("outside");
    fs::write(&outside, b"outside").unwrap();

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    let first = entries.first_mut().expect("manifest must contain entries");
    first["path"] = serde_json::Value::String("../outside".to_string());
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("parent traversal path should fail");
    assert!(
        err.to_string().contains("must not contain '..'"),
        "error should mention traversal rejection: {err}"
    );

    let _ = fs::remove_file(&outside);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn validate_manifest_rejects_absolute_path() {
    let base = temp_dir("manifest_absolute_path");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([15u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let abs = std::env::temp_dir().join(format!(
        "swarmtorch_absolute_target_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::write(&abs, b"abs").unwrap();

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    let first = entries.first_mut().expect("manifest must contain entries");
    first["path"] = serde_json::Value::String(abs.to_string_lossy().to_string());
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("absolute path should fail");
    assert!(
        err.to_string().contains("must be relative"),
        "error should mention relative path requirement: {err}"
    );

    let _ = fs::remove_file(&abs);
    let _ = fs::remove_dir_all(&base);
}

#[cfg(windows)]
#[test]
fn validate_manifest_rejects_windows_drive_absolute_path() {
    let base = temp_dir("manifest_windows_drive_absolute");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([19u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    let first = entries.first_mut().expect("manifest must contain entries");
    first["path"] = serde_json::Value::String("C:\\Windows\\System32\\drivers\\etc\\hosts".into());
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("windows drive absolute path should fail");
    assert!(
        err.to_string().contains("must be relative"),
        "error should mention relative path requirement: {err}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[cfg(windows)]
#[test]
fn validate_manifest_rejects_windows_verbatim_prefix_path() {
    let base = temp_dir("manifest_windows_verbatim_prefix");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([20u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    let first = entries.first_mut().expect("manifest must contain entries");
    first["path"] =
        serde_json::Value::String("\\\\?\\C:\\Windows\\System32\\drivers\\etc\\hosts".into());
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("windows verbatim-prefix path should fail");
    assert!(
        err.to_string().contains("must be relative"),
        "error should mention relative path requirement: {err}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[cfg(windows)]
#[test]
fn validate_manifest_rejects_symlink_escape_windows() {
    use sha2::{Digest, Sha256};
    use std::os::windows::fs::symlink_file;

    let base = temp_dir("manifest_symlink_escape_windows");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([21u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let outside = base.join("outside_target_windows.bin");
    fs::write(&outside, b"outside-data").unwrap();
    let symlink_path = bundle.run_dir().join("datasets").join("escape_link_win");
    match symlink_file(&outside, &symlink_path) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::PermissionDenied => {
            // Some Windows runners require elevated privileges for symlink creation.
            // In those environments, absolute/prefix traversal tests still exercise
            // path-confinement behavior, and this symlink-specific assertion is skipped.
            let _ = fs::remove_file(&outside);
            let _ = fs::remove_dir_all(&base);
            return;
        }
        Err(err) => panic!("windows symlink creation failed unexpectedly: {err}"),
    }

    let outside_bytes = fs::read(&outside).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(&outside_bytes);
    let outside_sha = format!("{:x}", hasher.finalize());
    let outside_len = outside_bytes.len() as u64;

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    let first = entries.first_mut().expect("manifest must contain entries");
    first["path"] = serde_json::Value::String("datasets/escape_link_win".to_string());
    first["sha256"] = serde_json::Value::String(outside_sha);
    first["bytes"] = serde_json::Value::from(outside_len);
    first["required"] = serde_json::Value::Bool(false);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("symlink escape should fail on windows");
    assert!(
        err.to_string().contains("escapes bundle root"),
        "error should mention bundle root escape: {err}"
    );

    let _ = fs::remove_file(&outside);
    let _ = fs::remove_file(&symlink_path);
    let _ = fs::remove_dir_all(&base);
}

#[cfg(unix)]
#[test]
fn validate_manifest_rejects_symlink_escape() {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::symlink;

    let base = temp_dir("manifest_symlink_escape");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([16u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let outside = base.join("outside_target.bin");
    fs::write(&outside, b"outside-data").unwrap();
    let symlink_path = bundle.run_dir().join("datasets").join("escape_link");
    symlink(&outside, &symlink_path).expect("symlink creation should succeed");

    let outside_bytes = fs::read(&outside).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(&outside_bytes);
    let outside_sha = format!("{:x}", hasher.finalize());
    let outside_len = outside_bytes.len() as u64;

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    let first = entries.first_mut().expect("manifest must contain entries");
    first["path"] = serde_json::Value::String("datasets/escape_link".to_string());
    first["sha256"] = serde_json::Value::String(outside_sha);
    first["bytes"] = serde_json::Value::from(outside_len);
    first["required"] = serde_json::Value::Bool(false);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("symlink escape should fail");
    assert!(
        err.to_string().contains("escapes bundle root"),
        "error should mention bundle root escape: {err}"
    );

    let _ = fs::remove_file(&outside);
    let _ = fs::remove_file(&symlink_path);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn validate_manifest_accepts_legitimate_nested_path() {
    let base = temp_dir("manifest_legitimate_nested");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([17u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    bundle
        .validate_manifest()
        .expect("valid nested bundle paths should pass");

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn validate_manifest_requires_registry_updates_ndjson() {
    let base = temp_dir("manifest_requires_registry_updates");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([18u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let entries = manifest
        .get_mut("entries")
        .and_then(serde_json::Value::as_array_mut)
        .expect("manifest entries must exist");
    entries.retain(|entry| {
        entry.get("path")
            != Some(&serde_json::Value::String(
                "datasets/registry_updates.ndjson".into(),
            ))
    });
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .validate_manifest()
        .expect_err("registry_updates.ndjson should be required");
    assert!(
        err.to_string()
            .contains("required manifest entry missing: datasets/registry_updates.ndjson"),
        "error should mention missing required update log: {err}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn emit_span_rejects_oversized_name_at_write_time() {
    let base = temp_dir("emit_span_rejects_oversized_name");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([111u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = RunArtifactSink::new(bundle);

    let span = SpanRecord {
        schema_version: 1,
        trace_id: TraceId::from_bytes([1u8; 16]),
        span_id: SpanId::from_bytes([1u8; 8]),
        parent_span_id: None,
        name: "x".repeat(MAX_RECORD_NAME_LEN + 1),
        start_unix_nanos: 1,
        end_unix_nanos: Some(2),
        attrs: AttrMap::new(),
    };

    let err = sink
        .emit_span(&span)
        .expect_err("oversized span name should be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn emit_event_rejects_too_many_attrs_at_write_time() {
    let base = temp_dir("emit_event_rejects_too_many_attrs");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([112u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = RunArtifactSink::new(bundle);

    let mut attrs = AttrMap::new();
    for i in 0..(MAX_RECORD_ATTRS + 1) {
        attrs.insert(format!("k{i}"), AttrValue::Bool(true));
    }
    let event = EventRecord {
        schema_version: 1,
        ts_unix_nanos: 1,
        trace_id: TraceId::from_bytes([1u8; 16]),
        span_id: None,
        name: "evt".to_string(),
        attrs,
    };

    let err = sink
        .emit_event(&event)
        .expect_err("too many attrs should be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn emit_metric_accepts_valid_record_at_write_time() {
    let base = temp_dir("emit_metric_accepts_valid");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([113u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = RunArtifactSink::new(bundle);

    let metric = MetricRecord {
        schema_version: 1,
        ts_unix_nanos: 1,
        trace_id: TraceId::from_bytes([1u8; 16]),
        span_id: None,
        name: "m".to_string(),
        value: 1.0,
        unit: Some("count".to_string()),
        attrs: AttrMap::new(),
    };

    sink.emit_metric(&metric)
        .expect("valid metric should be accepted");

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn emit_metric_rejects_oversized_unit_at_write_time() {
    let base = temp_dir("emit_metric_rejects_oversized_unit");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([114u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = RunArtifactSink::new(bundle);

    let metric = MetricRecord {
        schema_version: 1,
        ts_unix_nanos: 1,
        trace_id: TraceId::from_bytes([1u8; 16]),
        span_id: None,
        name: "m".to_string(),
        value: 1.0,
        unit: Some("x".repeat(MAX_METRIC_UNIT_LEN + 1)),
        attrs: AttrMap::new(),
    };

    let err = sink
        .emit_metric(&metric)
        .expect_err("oversized metric unit should be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

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
        execution_hint: None,
        cache_policy: None,
        materialization_policy: None,
        resources: None,
        op_hash: None,
    });

    bundle.write_graph(&graph).unwrap();

    let on_disk: GraphV1 = read_json(&bundle.run_dir().join("graph.json")).unwrap();
    assert_eq!(on_disk.schema_version, 1);
    assert_eq!(on_disk.nodes.len(), 1);

    let node = &on_disk.nodes[0];
    assert!(node.node_id.is_some());
    assert!(node.node_def_hash.is_some());
    assert!(node.code_ref.is_some());

    // Verify derived values match the core hashing rules.
    let expected_id = node_id_from_key(&node.node_key).to_string();
    assert_eq!(node.node_id.unwrap().to_string(), expected_id);

    let digest = node_def_hash_v1(node).unwrap();
    let expected_hash = hex_lower(&digest);
    assert_eq!(node.node_def_hash.as_ref().unwrap(), &expected_hash);

    let _ = fs::remove_dir_all(&base);
}

/// M-09: write_graph must reject nodes that violate field-bounds validation.
#[test]
fn write_graph_rejects_oversized_node_key() {
    let base = temp_dir("write_graph_rejects_oversized");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([10u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let mut graph = GraphV1::default();
    graph.nodes.push(NodeV1 {
        node_key: "x".repeat(MAX_NODE_KEY_LEN + 1),
        node_id: None,
        op_kind: OpKind::Data,
        op_type: "validate".to_string(),
        inputs: vec![],
        outputs: vec![],
        params: CanonParams::new(),
        code_ref: None,
        unsafe_surface: false,
        execution_trust: ExecutionTrust::Core,
        node_def_hash: None,
        execution_hint: None,
        cache_policy: None,
        materialization_policy: None,
        resources: None,
        op_hash: None,
    });

    let result = bundle.write_graph(&graph);
    assert!(
        result.is_err(),
        "write_graph must reject oversized node_key"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("node_key length"),
        "error should mention node_key length: {err_msg}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn write_graph_accepts_node_with_execution_hint() {
    let base = temp_dir("write_graph_accepts_execution_hint");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([55u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let mut graph = GraphV1::default();
    graph.nodes.push(NodeV1 {
        node_key: "prep/hinted".to_string(),
        node_id: None,
        op_kind: OpKind::Data,
        op_type: "validate".to_string(),
        inputs: vec![],
        outputs: vec![],
        params: CanonParams::new(),
        code_ref: None,
        unsafe_surface: false,
        execution_trust: ExecutionTrust::Core,
        node_def_hash: None,
        execution_hint: Some(ExecutionHint {
            preferred_profile: Some(PreferredProfile::EdgeStd),
            device_affinity: Some(DeviceAffinity::Coordinator),
            memory_budget_bytes: Some(65_536),
        }),
        cache_policy: None,
        materialization_policy: None,
        resources: None,
        op_hash: None,
    });

    bundle
        .write_graph(&graph)
        .expect("execution_hint should be accepted on write path");
    let on_disk: GraphV1 = read_json(&bundle.run_dir().join("graph.json")).unwrap();
    assert_eq!(on_disk.nodes.len(), 1);
    assert!(on_disk.nodes[0].execution_hint.is_some());

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
        execution_hint: None,
        cache_policy: None,
        materialization_policy: None,
        resources: None,
        op_hash: None,
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
        execution_hint: None,
        cache_policy: None,
        materialization_policy: None,
        resources: None,
        op_hash: None,
    }
}

#[test]
fn materialize_multi_output_is_all_or_nothing_on_write_failure() {
    let base = temp_dir("materialize_all_or_nothing_failure");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([14u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    let source = SourceDescriptorV0 {
        uri: "s3://bucket/raw".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: None,
    };
    let ingest = make_source_node("ingest/raw");
    session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            source,
            None,
            &ingest,
        )
        .unwrap();

    // Force write-path failure before any staged commit can complete.
    let registry_updates_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("registry_updates.ndjson");
    fs::remove_file(&registry_updates_path).unwrap();
    fs::create_dir_all(&registry_updates_path).unwrap();

    let node = make_transform_node(
        "transform/fanout",
        &["dataset://ns/raw"],
        &["dataset://ns/out_a", "dataset://ns/out_b"],
        ExecutionTrust::Core,
    );
    let err = session
        .materialize_node_outputs(
            &node,
            &[
                OutputSpec {
                    asset_key: "dataset://ns/out_a".to_string(),
                    schema: None,
                    rows: Some(1),
                    bytes: Some(1),
                },
                OutputSpec {
                    asset_key: "dataset://ns/out_b".to_string(),
                    schema: None,
                    rows: Some(1),
                    bytes: Some(1),
                },
            ],
            1000,
            CacheDecisionV0::Miss,
            1,
        )
        .expect_err("write failure should bubble");
    assert!(
        err.to_string().contains("Is a directory")
            || err.to_string().contains("is a directory")
            || err.kind() == io::ErrorKind::PermissionDenied
            || err.kind() == io::ErrorKind::Other
    );
    assert!(session.fingerprint("dataset://ns/out_a").is_none());
    assert!(session.fingerprint("dataset://ns/out_b").is_none());

    let materializations = fs::read_to_string(
        sink.bundle()
            .run_dir()
            .join("datasets")
            .join("materializations.ndjson"),
    )
    .unwrap();
    assert!(
        materializations.trim().is_empty(),
        "no partial materialization lines should be persisted when first staged write fails"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn materialize_failure_preserves_in_memory_session_state() {
    let base = temp_dir("materialize_failure_preserves_state");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([15u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    let source = SourceDescriptorV0 {
        uri: "s3://bucket/raw".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: None,
    };
    let ingest = make_source_node("ingest/raw");
    session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            source,
            None,
            &ingest,
        )
        .unwrap();
    let source_fp_before = session.fingerprint("dataset://ns/raw").unwrap().to_string();

    let registry_updates_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("registry_updates.ndjson");
    fs::remove_file(&registry_updates_path).unwrap();
    fs::create_dir_all(&registry_updates_path).unwrap();

    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    let failed = session.materialize_node_outputs(
        &node,
        &[OutputSpec {
            asset_key: "dataset://ns/clean".to_string(),
            schema: None,
            rows: Some(1),
            bytes: Some(1),
        }],
        2000,
        CacheDecisionV0::Miss,
        2,
    );
    assert!(failed.is_err());
    assert_eq!(
        session.fingerprint("dataset://ns/raw"),
        Some(source_fp_before.as_str())
    );
    assert!(session.fingerprint("dataset://ns/clean").is_none());

    // Restore write path and verify the same session can continue safely.
    fs::remove_dir_all(&registry_updates_path).unwrap();
    fs::File::create(&registry_updates_path).unwrap();
    session
        .materialize_node_outputs(
            &node,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(1),
                bytes: Some(1),
            }],
            3000,
            CacheDecisionV0::Miss,
            3,
        )
        .unwrap();
    assert!(session.fingerprint("dataset://ns/clean").is_some());
    assert_eq!(
        session.fingerprint("dataset://ns/raw"),
        Some(source_fp_before.as_str())
    );

    let _ = fs::remove_dir_all(&base);
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
fn unsafe_reasons_include_untrusted_input() {
    let base = temp_dir("unsafe_reasons_untrusted_input");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([51u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

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

    let mat_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("materializations.ndjson");
    let content = fs::read_to_string(&mat_path).unwrap();
    let rows: Vec<MaterializationRecordCompat> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let row = rows[0].clone().into_v2();

    assert!(row.unsafe_surface, "untrusted input should mark unsafe");
    assert!(
        row.unsafe_reasons.contains(&UnsafeReasonV0::UntrustedInput),
        "unsafe reasons should include UntrustedInput"
    );
    assert!(
        !row.unsafe_reasons
            .contains(&UnsafeReasonV0::UnsafeExtension),
        "core execution should not include UnsafeExtension reason"
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

#[test]
fn unsafe_reasons_include_unsafe_extension() {
    let base = temp_dir("unsafe_reasons_unsafe_extension");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([61u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

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

    let mat_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("materializations.ndjson");
    let content = fs::read_to_string(&mat_path).unwrap();
    let rows: Vec<MaterializationRecordCompat> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let row = rows[0].clone().into_v2();

    assert!(row.unsafe_surface, "unsafe extension should mark unsafe");
    assert!(
        row.unsafe_reasons
            .contains(&UnsafeReasonV0::UnsafeExtension),
        "unsafe reasons should include UnsafeExtension"
    );
    assert!(
        !row.unsafe_reasons.contains(&UnsafeReasonV0::UntrustedInput),
        "trusted input should not include UntrustedInput reason"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn untrusted_transform_audit_adds_unsafe_extension_reason() {
    let base = temp_dir("unsafe_reasons_transform_audit");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([62u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

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

    session.record_transform_applied(&TransformAuditV0 {
        transform_name: "dp_clip".to_string(),
        core_trusted: false,
        round_id: 1,
    });

    let transform = make_transform_node(
        "transform/core_with_transform",
        &["dataset://ns/trusted"],
        &["dataset://ns/out"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &transform,
            &[OutputSpec {
                asset_key: "dataset://ns/out".to_string(),
                schema: None,
                rows: Some(10),
                bytes: Some(100),
            }],
            1000,
            false,
            10,
        )
        .unwrap();

    let mat_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("materializations.ndjson");
    let content = fs::read_to_string(&mat_path).unwrap();
    let rows: Vec<MaterializationRecordCompat> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let row = rows[0].clone().into_v2();

    assert!(row.unsafe_surface);
    assert!(row
        .unsafe_reasons
        .contains(&UnsafeReasonV0::UnsafeExtension));
    assert_eq!(row.applied_transforms.len(), 1);
    assert_eq!(row.applied_transforms[0].transform_name, "dp_clip");
    assert!(!row.applied_transforms[0].core_trusted);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn source_descriptor_rejects_oversized_uri() {
    let base = temp_dir("source_descriptor_oversized_uri");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([62u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    let source = SourceDescriptorV0 {
        uri: format!("s3://bucket/{}", "a".repeat(MAX_SOURCE_URI_LEN + 1)),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: None,
    };
    let ingest = make_source_node("ingest/s3");

    let err = session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            source,
            None,
            &ingest,
        )
        .expect_err("oversized uri should be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains("uri too long"),
        "error should mention uri length violation: {}",
        err
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn source_descriptor_rejects_oversized_etag_or_version() {
    let base = temp_dir("source_descriptor_oversized_etag");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([63u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    let source = SourceDescriptorV0 {
        uri: "s3://bucket/path".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: Some("v".repeat(MAX_ETAG_OR_VERSION_LEN + 1)),
    };
    let ingest = make_source_node("ingest/s3");

    let err = session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            source,
            None,
            &ingest,
        )
        .expect_err("oversized etag_or_version should be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains("etag_or_version too long"),
        "error should mention etag_or_version length violation: {}",
        err
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
fn materialize_fails_on_invalid_input_fingerprint_hex() {
    let base = temp_dir("invalid_input_fingerprint");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([71u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    // Register source, then intentionally corrupt fingerprint.
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
    session
        .test_registry_entry_mut("dataset://ns/raw")
        .expect("test setup must register dataset://ns/raw")
        .fingerprint_v0 = "zzzz".to_string();

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

    assert!(result.is_err(), "should fail on invalid input fingerprint");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("invalid fingerprint"),
        "error should mention invalid fingerprint: {err_msg}"
    );
    assert!(
        err_msg.contains("dataset://ns/raw"),
        "error should name the invalid input asset: {err_msg}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn lineage_uses_pre_mutation_input_fingerprint_for_in_place_transform() {
    // Verify that lineage edges reference the INPUT fingerprint from BEFORE
    // the output was inserted into the registry. This prevents corruption
    // when an output asset_key matches an input asset_key (re-materialization).
    let base = temp_dir("lineage_premutation");
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

    // Capture the source fingerprint BEFORE transform
    let source_fp = session.fingerprint("dataset://ns/raw").unwrap().to_string();

    // In-place transform: raw -> raw
    let transform = make_transform_node(
        "transform/recompute_raw",
        &["dataset://ns/raw"],
        &["dataset://ns/raw"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &transform,
            &[OutputSpec {
                asset_key: "dataset://ns/raw".to_string(),
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
    assert_ne!(
        lineage.edges[0].output_fingerprint_v0, source_fp,
        "in-place transform output fingerprint should reflect new materialization state"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn materialize_rejects_output_not_declared_in_node() {
    let base = temp_dir("undeclared_output");
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
fn materialize_rejects_missing_declared_node_output() {
    let base = temp_dir("missing_declared_output");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([74u8; 16]);
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

    // Node declares two outputs; materialization provides only one.
    let transform = make_transform_node(
        "transform/split",
        &["dataset://ns/raw"],
        &["dataset://ns/out_a", "dataset://ns/out_b"],
        ExecutionTrust::Core,
    );
    let result = session.materialize_node_outputs(
        &transform,
        &[OutputSpec {
            asset_key: "dataset://ns/out_a".to_string(),
            schema: None,
            rows: Some(10),
            bytes: Some(100),
        }],
        1000,
        false,
        50,
    );

    assert!(
        result.is_err(),
        "should reject missing declared node output"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("missing declared node output"),
        "error should mention missing declared output: {err_msg}"
    );
    assert!(
        err_msg.contains("dataset://ns/out_b"),
        "error should identify missing output key: {err_msg}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn materialize_rejects_duplicate_output_asset_keys() {
    let base = temp_dir("duplicate_output");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([75u8; 16]);
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

#[test]
fn materialize_rejects_duplicate_declared_node_outputs() {
    let base = temp_dir("duplicate_declared_output");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([76u8; 16]);
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

    // Node declares duplicate output keys.
    let mut transform = make_transform_node(
        "transform/dup_declared",
        &["dataset://ns/raw"],
        &["dataset://ns/out"],
        ExecutionTrust::Core,
    );
    transform
        .outputs
        .push(swarm_torch_core::run_graph::AssetRefV1 {
            asset_key: "dataset://ns/out".to_string(),
            fingerprint: None,
        });

    let result = session.materialize_node_outputs(
        &transform,
        &[OutputSpec {
            asset_key: "dataset://ns/out".to_string(),
            schema: None,
            rows: Some(10),
            bytes: Some(100),
        }],
        1000,
        false,
        50,
    );

    assert!(
        result.is_err(),
        "should reject duplicate declared output keys in node.outputs"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("duplicate node output asset_key"),
        "error should mention duplicate declared outputs: {err_msg}"
    );

    let _ = fs::remove_dir_all(&base);
}

// ── Phase 3: Cache-Hit Detection tests ──────────────────────────

#[test]
fn predict_err_on_missing_input() {
    let base = temp_dir("predict_missing");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([80u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let session = DataOpsSession::new(Arc::clone(&sink));

    // Node has an input that isn't registered
    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/missing"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    let result = session.predict(
        &node,
        &[OutputSpecCore {
            asset_key: "dataset://ns/clean".to_string(),
            schema: None,
        }],
    );

    assert!(result.is_err(), "predict should fail on missing input");
    match result.unwrap_err() {
        PredictError::MissingInput(key) => {
            assert_eq!(key, "dataset://ns/missing");
        }
        other => panic!("expected MissingInput, got {:?}", other),
    }

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn predict_err_on_output_contract_violation() {
    let base = temp_dir("predict_output_contract");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([83u8; 16]);
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

    // Node declares output out_a, but prediction asks for out_b.
    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/out_a"],
        ExecutionTrust::Core,
    );
    let result = session.predict(
        &node,
        &[OutputSpecCore {
            asset_key: "dataset://ns/out_b".to_string(),
            schema: None,
        }],
    );

    assert!(
        result.is_err(),
        "predict should fail on output contract violation"
    );
    match result.unwrap_err() {
        PredictError::OutputContract(msg) => {
            assert!(
                msg.contains("not declared"),
                "error should mention undeclared output: {msg}"
            );
            assert!(
                msg.contains("dataset://ns/out_b"),
                "error should include output key: {msg}"
            );
        }
        other => panic!("expected OutputContract, got {:?}", other),
    }

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn schema_aware_prediction_matches_materialization() {
    // Predict fingerprints then materialize — they must agree.
    let base = temp_dir("predict_matches_mat");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([81u8; 16]);
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

    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );

    // Predict (schema=None)
    let predicted = session
        .predict(
            &node,
            &[OutputSpecCore {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
            }],
        )
        .unwrap();

    assert_eq!(predicted.len(), 1);
    let predicted_fp = &predicted[0].fingerprint_v0;

    // Before materialization: no cache hit
    assert!(
        !session.is_cache_hit("dataset://ns/clean", predicted_fp),
        "should not be a cache hit before materialization"
    );

    // Materialize with the same node + outputs (schema=None)
    session
        .materialize_node_outputs(
            &node,
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

    // After materialization: fingerprints must match
    let actual_fp = session
        .fingerprint("dataset://ns/clean")
        .unwrap()
        .to_string();
    assert_eq!(
        *predicted_fp, actual_fp,
        "predicted fingerprint must match materialized fingerprint"
    );

    // Now it IS a cache hit
    assert!(
        session.is_cache_hit("dataset://ns/clean", predicted_fp),
        "should be a cache hit after materialization"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn cache_hit_is_asset_key_scoped() {
    let base = temp_dir("cache_hit_scoped");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([82u8; 16]);
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

    // Materialize output A
    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/out_a"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &node,
            &[OutputSpec {
                asset_key: "dataset://ns/out_a".to_string(),
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
        .fingerprint("dataset://ns/out_a")
        .unwrap()
        .to_string();

    // Asset A's fingerprint is NOT a cache hit for asset B
    assert!(
        !session.is_cache_hit("dataset://ns/out_b", &fp_a),
        "cache hit must be scoped by asset_key, not just fingerprint"
    );

    // But it IS for asset A
    assert!(
        session.is_cache_hit("dataset://ns/out_a", &fp_a),
        "same asset_key + same fingerprint should be a cache hit"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn materialization_v2_includes_input_provenance() {
    let base = temp_dir("mat_v2_input_provenance");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([90u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

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
    let in_fp = session.fingerprint("dataset://ns/raw").unwrap().to_string();

    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &node,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(12),
                bytes: Some(345),
            }],
            1000,
            false,
            25,
        )
        .unwrap();

    let mat_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("materializations.ndjson");
    let content = fs::read_to_string(&mat_path).unwrap();
    let rows: Vec<MaterializationRecordCompat> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    assert_eq!(rows.len(), 1, "expected one materialization row");
    let row_v2 = rows[0].clone().into_v2();
    assert_eq!(row_v2.schema_version, MATERIALIZATION_SCHEMA_V2);
    assert_eq!(
        row_v2.input_asset_keys,
        vec!["dataset://ns/raw".to_string()],
        "v2 row must include input asset keys"
    );
    assert_eq!(
        row_v2.input_fingerprints_v0,
        vec![in_fp],
        "v2 row must include input fingerprints"
    );
    assert_eq!(row_v2.op_type, "transform");
    assert_eq!(row_v2.status, MaterializationStatusV0::Ok);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn cache_hit_is_derived_from_cache_decision() {
    let base = temp_dir("cache_hit_derived_from_decision");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([91u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

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

    let node = make_transform_node(
        "transform/cache",
        &["dataset://ns/raw"],
        &["dataset://ns/out_a", "dataset://ns/out_b"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &node,
            &[
                OutputSpec {
                    asset_key: "dataset://ns/out_a".to_string(),
                    schema: None,
                    rows: Some(1),
                    bytes: Some(1),
                },
                OutputSpec {
                    asset_key: "dataset://ns/out_b".to_string(),
                    schema: None,
                    rows: Some(1),
                    bytes: Some(1),
                },
            ],
            1000,
            true,
            5,
        )
        .unwrap();

    let mat_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("materializations.ndjson");
    let content = fs::read_to_string(&mat_path).unwrap();
    let rows: Vec<_> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<MaterializationRecordCompat>(line)
                .unwrap()
                .into_v2()
        })
        .collect();

    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row.cache_decision, CacheDecisionV0::Hit);
        assert_eq!(row.cache_hit, Some(true));
        assert!(
            row.cache_key_v0.as_ref().is_some_and(|v| v.len() == 64),
            "cache_key_v0 should be a 64-char hex digest"
        );
    }

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn cache_key_v0_stable_for_same_inputs() {
    let node = make_transform_node(
        "transform/cache_key",
        &["dataset://ns/raw"],
        &["dataset://ns/out"],
        ExecutionTrust::Core,
    );
    let upstream = vec![[7u8; 32], [11u8; 32]];
    let k1 = cache_key_v0(&node, &upstream, "core");
    let k2 = cache_key_v0(&node, &upstream, "core");
    assert_eq!(k1, k2, "cache_key_v0 must be deterministic");
}

#[test]
fn cache_key_v0_changes_when_input_fingerprint_changes() {
    let node = make_transform_node(
        "transform/cache_key_change",
        &["dataset://ns/raw"],
        &["dataset://ns/out"],
        ExecutionTrust::Core,
    );
    let a = vec![[7u8; 32], [11u8; 32]];
    let b = vec![[8u8; 32], [11u8; 32]];
    let k1 = cache_key_v0(&node, &a, "core");
    let k2 = cache_key_v0(&node, &b, "core");
    assert_ne!(
        k1, k2,
        "cache_key_v0 must change when upstream fingerprints change"
    );
}

#[test]
fn registry_updates_log_replays_to_snapshot_equivalence() {
    let base = temp_dir("registry_updates_replay_equivalence");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([101u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session =
        DataOpsSession::with_profile(Arc::clone(&sink), SnapshotProfile::streaming(2));

    let source = SourceDescriptorV0 {
        uri: "s3://bucket/data".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: Some("v1".to_string()),
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

    let clean = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &clean,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(10),
                bytes: Some(100),
            }],
            1000,
            false,
            10,
        )
        .unwrap();

    // In-place rewrite to ensure replay handles "latest wins" by asset key.
    let rewrite = make_transform_node(
        "transform/rewrite",
        &["dataset://ns/clean"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &rewrite,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(10),
                bytes: Some(120),
            }],
            2000,
            false,
            12,
        )
        .unwrap();

    session.finalize().unwrap();

    let snapshot: DatasetRegistryV1 = read_json(
        &sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("registry.json"),
    )
    .unwrap();
    let updates: Vec<DatasetEntryV1> = fs::read_to_string(
        sink.bundle()
            .run_dir()
            .join("datasets")
            .join("registry_updates.ndjson"),
    )
    .unwrap()
    .lines()
    .filter(|line| !line.trim().is_empty())
    .map(|line| serde_json::from_str::<DatasetEntryV1>(line).unwrap())
    .collect();

    let mut replayed = BTreeMap::new();
    for entry in updates {
        replayed.insert(entry.asset_key.clone(), entry);
    }
    let mut snapshot_map = BTreeMap::new();
    for entry in snapshot.datasets {
        snapshot_map.insert(entry.asset_key.clone(), entry);
    }

    assert_eq!(
        replayed, snapshot_map,
        "replayed registry updates must match compacted snapshot"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn lineage_updates_log_replays_to_snapshot_equivalence() {
    let base = temp_dir("lineage_updates_replay_equivalence");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([102u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session =
        DataOpsSession::with_profile(Arc::clone(&sink), SnapshotProfile::streaming(3));

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

    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &node,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(10),
                bytes: Some(100),
            }],
            1000,
            false,
            10,
        )
        .unwrap();
    // Same transform/output again: should dedupe lineage edges.
    session
        .materialize_node_outputs(
            &node,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(10),
                bytes: Some(100),
            }],
            2000,
            false,
            10,
        )
        .unwrap();

    session.finalize().unwrap();

    let snapshot: DatasetLineageV1 = read_json(
        &sink
            .bundle()
            .run_dir()
            .join("datasets")
            .join("lineage.json"),
    )
    .unwrap();
    let updates: Vec<LineageEdgeV1> = fs::read_to_string(
        sink.bundle()
            .run_dir()
            .join("datasets")
            .join("lineage_edges.ndjson"),
    )
    .unwrap()
    .lines()
    .filter(|line| !line.trim().is_empty())
    .map(|line| serde_json::from_str::<LineageEdgeV1>(line).unwrap())
    .collect();

    let mut replayed = BTreeMap::new();
    for edge in updates {
        let key = (
            edge.input_fingerprint_v0.clone(),
            edge.output_fingerprint_v0.clone(),
            edge.node_id.to_string(),
        );
        replayed.insert(key, edge);
    }
    let mut snapshot_map = BTreeMap::new();
    for edge in snapshot.edges {
        let key = (
            edge.input_fingerprint_v0.clone(),
            edge.output_fingerprint_v0.clone(),
            edge.node_id.to_string(),
        );
        snapshot_map.insert(key, edge);
    }

    assert_eq!(
        replayed, snapshot_map,
        "replayed lineage updates must match compacted snapshot"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn streaming_profile_defers_snapshot_rewrite_until_interval() {
    let base = temp_dir("streaming_profile_defers_snapshot");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([103u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session =
        DataOpsSession::with_profile(Arc::clone(&sink), SnapshotProfile::streaming(2));

    let registry_path = sink
        .bundle()
        .run_dir()
        .join("datasets")
        .join("registry.json");
    let before = fs::read_to_string(&registry_path).unwrap();

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

    let after_first = fs::read_to_string(&registry_path).unwrap();
    assert_eq!(
        before, after_first,
        "streaming profile should defer snapshot rewrite before interval"
    );

    let updates = fs::read_to_string(
        sink.bundle()
            .run_dir()
            .join("datasets")
            .join("registry_updates.ndjson"),
    )
    .unwrap();
    assert_eq!(
        updates
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count(),
        1,
        "first mutation should still append one registry update"
    );

    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &node,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(1),
                bytes: Some(1),
            }],
            1000,
            false,
            1,
        )
        .unwrap();

    let after_second: DatasetRegistryV1 = read_json(&registry_path).unwrap();
    assert!(
        after_second.datasets.len() >= 2,
        "second mutation should trigger snapshot compaction at interval=2"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn strict_profile_updates_manifest_on_each_write() {
    let base = temp_dir("strict_profile_manifest_always");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([104u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let profile = ArtifactWriteProfile {
        snapshot_profile: SnapshotProfile::strict(),
        manifest_policy: ManifestRefreshPolicy::Always,
    };
    let sink = Arc::new(RunArtifactSink::with_profile(bundle, profile));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

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

    assert!(
        sink.validate_manifest().is_ok(),
        "manifest should refresh after each write under Always policy"
    );

    let node = make_transform_node(
        "transform/clean",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::Core,
    );
    session
        .materialize_node_outputs(
            &node,
            &[OutputSpec {
                asset_key: "dataset://ns/clean".to_string(),
                schema: None,
                rows: Some(1),
                bytes: Some(1),
            }],
            1000,
            false,
            1,
        )
        .unwrap();

    assert!(
        sink.validate_manifest().is_ok(),
        "manifest should remain valid mid-run under Always policy"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn manifest_interval_policy_refreshes_after_n_writes() {
    let base = temp_dir("manifest_interval_policy");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([105u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let profile = ArtifactWriteProfile {
        snapshot_profile: SnapshotProfile::strict(),
        manifest_policy: ManifestRefreshPolicy::IntervalN(2),
    };
    let sink = RunArtifactSink::with_profile(bundle, profile);

    let event = EventRecord {
        schema_version: 1,
        trace_id: TraceId::from_bytes([1u8; 16]),
        span_id: None,
        name: "event/a".to_string(),
        ts_unix_nanos: 1000,
        attrs: AttrMap::new(),
    };
    sink.append_event(&event).unwrap();
    assert!(
        sink.validate_manifest().is_err(),
        "manifest should be stale after first write when interval is 2"
    );

    let event_b = EventRecord {
        schema_version: 1,
        trace_id: TraceId::from_bytes([1u8; 16]),
        span_id: None,
        name: "event/b".to_string(),
        ts_unix_nanos: 2000,
        attrs: AttrMap::new(),
    };
    sink.append_event(&event_b).unwrap();
    assert!(
        sink.validate_manifest().is_ok(),
        "manifest should refresh after second write when interval is 2"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn artifact_pipeline_smoke_after_artifacts_module_split() {
    let base = temp_dir("artifact_pipeline_smoke_split");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([106u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    let source = SourceDescriptorV0 {
        uri: "s3://bucket/raw.parquet".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: Some("v1".to_string()),
    };
    let ingest = make_source_node("ingest/raw");
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
                rows: Some(10),
                bytes: Some(100),
            }],
            1_000,
            false,
            5,
        )
        .unwrap();

    session.finalize().unwrap();
    sink.validate_manifest().unwrap();
    assert!(
        sink.bundle()
            .run_dir()
            .join("datasets")
            .join("materializations.ndjson")
            .exists(),
        "materializations.ndjson should exist after smoke pipeline"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn materialization_schema_stable_after_artifacts_module_split() {
    let base = temp_dir("materialization_schema_stable_split");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([107u8; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let sink = Arc::new(RunArtifactSink::new(bundle));
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    let source = SourceDescriptorV0 {
        uri: "s3://bucket/raw.parquet".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
        etag_or_version: Some("v1".to_string()),
    };
    let ingest = make_source_node("ingest/raw");
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
                rows: Some(10),
                bytes: Some(100),
            }],
            1_000,
            false,
            5,
        )
        .unwrap();
    session.finalize().unwrap();

    let content = fs::read_to_string(
        sink.bundle()
            .run_dir()
            .join("datasets")
            .join("materializations.ndjson"),
    )
    .unwrap();
    let rows: Vec<MaterializationRecordCompat> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    let row = rows[0].clone().into_v2();
    assert_eq!(row.schema_version, MATERIALIZATION_SCHEMA_V2);
    assert_eq!(row.asset_key, "dataset://ns/clean");
    assert_eq!(row.op_type, "transform");
    assert_eq!(row.input_asset_keys, vec!["dataset://ns/raw".to_string()]);

    let _ = fs::remove_dir_all(&base);
}

// -------------------------------------------------------------------------
// Wave 10 Slice 10A (W10-02): BundleLockGuard, ManifestSignatureV1, TrustPolicy
// -------------------------------------------------------------------------

#[test]
fn bundle_lock_guard_rejects_concurrent_acquisition_with_bundle_busy() {
    let base = temp_dir("bundle_lock_contention");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0x61; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let guard_1 = bundle
        .acquire_lock(LockAcquisitionPolicy::FailFast)
        .expect("first lock acquisition should succeed");
    assert!(guard_1.lock_path().exists());

    let err = BundleLockGuard::acquire(bundle.run_dir(), LockAcquisitionPolicy::FailFast)
        .expect_err("second concurrent lock acquisition must fail closed");
    assert!(
        matches!(err, BundleLockError::BundleBusy { .. }),
        "expected BundleBusy error, got: {err:?}"
    );

    drop(guard_1);
    let guard_2 = bundle
        .acquire_lock(LockAcquisitionPolicy::FailFast)
        .expect("lock acquisition after drop should succeed");
    drop(guard_2);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn validate_manifest_strict_verifies_ed25519_signature_and_rejects_tampering() {
    use swarm_torch_core::crypto::KeyPair;

    let base = temp_dir("manifest_strict_signature");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0x62; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let signer = KeyPair::from_seed([0x77; 32]).unwrap();
    let untrusted = KeyPair::from_seed([0x88; 32]).unwrap();

    let signed_manifest = bundle
        .finalize_signed_manifest(&signer)
        .expect("signing manifest should succeed");
    assert!(signed_manifest.signature.is_some());

    // 1. Strict policy with signer's public key succeeds.
    let strict_ok = TrustPolicy::Strict {
        trusted_public_keys: vec![*signer.public_key()],
        expected_run_id: Some(run_id),
        min_signed_unix_nanos: Some(1),
    };
    let verified = bundle
        .verify_and_load(&strict_ok)
        .expect("strict verification with trusted signer should succeed");
    assert!(matches!(
        verified.status(),
        VerificationStatus::VerifiedStrict { .. }
    ));

    // 2. Strict policy with untrusted public key fails closed.
    let strict_untrusted = TrustPolicy::Strict {
        trusted_public_keys: vec![*untrusted.public_key()],
        expected_run_id: Some(run_id),
        min_signed_unix_nanos: None,
    };
    let err = bundle
        .verify_and_load(&strict_untrusted)
        .expect_err("untrusted signer key must be rejected");
    assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);

    // 3. Tamper with an entry's required flag in manifest.json without re-signing -> fails closed.
    let manifest_path = bundle.run_dir().join("manifest.json");
    let mut raw_manifest: ManifestV1 = read_json(&manifest_path).unwrap();
    raw_manifest.entries[0].required = !raw_manifest.entries[0].required;
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&raw_manifest).unwrap(),
    )
    .unwrap();

    let err = bundle
        .verify_and_load(&strict_ok)
        .expect_err("tampered manifest entry must fail signature verification");
    assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn validate_manifest_strict_rejects_unsigned_manifest() {
    use swarm_torch_core::crypto::KeyPair;

    let base = temp_dir("manifest_strict_rejects_unsigned");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0x63; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    bundle.finalize_manifest().unwrap();

    let signer = KeyPair::from_seed([0x79; 32]).unwrap();
    let strict_policy = TrustPolicy::Strict {
        trusted_public_keys: vec![*signer.public_key()],
        expected_run_id: Some(run_id),
        min_signed_unix_nanos: None,
    };

    let err = bundle
        .verify_and_load(&strict_policy)
        .expect_err("unsigned manifest must be rejected under TrustPolicy::Strict");
    assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);

    // LegacyPermissive still accepts unsigned manifest with Explicit status.
    let permissive = bundle
        .verify_and_load(&TrustPolicy::LegacyPermissive)
        .expect("LegacyPermissive should accept unsigned manifest");
    assert_eq!(
        permissive.status(),
        VerificationStatus::UnverifiedLegacyPermissive
    );

    let _ = fs::remove_dir_all(&base);
}

// -------------------------------------------------------------------------
// Wave 13 Phase 13B-2 (ST-05): Kernel Locks, Checked NDJSON, & CAS Integrity
// -------------------------------------------------------------------------

#[test]
fn st05_persistent_bundle_lock_preserves_inode_and_survives_process_crash_and_contention() {
    use std::io::{BufRead, Write};
    use std::process::{Command, Stdio};
    use std::time::Duration;

    if let Ok(run_dir) = std::env::var("SWARM_TORCH_LOCK_CHILD_RUN_DIR") {
        let _guard = BundleLockGuard::acquire(
            std::path::Path::new(&run_dir),
            LockAcquisitionPolicy::FailFast,
        )
        .expect("child process should acquire persistent .bundle.lock");
        println!("LOCKED");
        io::stdout().flush().unwrap();
        std::thread::sleep(Duration::from_secs(30));
        return;
    }

    let base = temp_dir("st05_kernel_lock_crash");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0xB1; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let lock_path = bundle.run_dir().join(BUNDLE_LOCK_FILE_NAME);
    assert!(
        lock_path.exists(),
        "persistent .bundle.lock must remain on disk after RunArtifactBundle::create drops its lock"
    );

    // Pre-existing junk in unlocked .bundle.lock must be replaced only after locking.
    fs::write(&lock_path, b"stale-corrupt-lock-bytes").unwrap();
    let g1 = bundle
        .acquire_lock(LockAcquisitionPolicy::FailFast)
        .expect("locking over persistent inode must succeed");
    let g1_nonce = g1.lease_nonce();
    drop(g1);
    assert!(
        lock_path.exists(),
        "Drop must unlock without unlinking persistent .bundle.lock"
    );
    let lock_json: serde_json::Value = read_json(&lock_path).unwrap();
    assert_eq!(lock_json["schema_version"].as_u64(), Some(1));
    assert_eq!(lock_json["lease_nonce"].as_u64(), Some(g1_nonce));

    // Real cross-process exclusive contention + forced child crash (SIGKILL / TerminateProcess).
    let exe = std::env::current_exe().expect("locate current test executable");
    let mut child = Command::new(exe)
        .arg("--exact")
        .arg("artifacts::tests::st05_persistent_bundle_lock_preserves_inode_and_survives_process_crash_and_contention")
        .arg("--nocapture")
        .env("SWARM_TORCH_LOCK_CHILD_RUN_DIR", bundle.run_dir())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn child process lock holder");

    let stdout = child.stdout.take().expect("child stdout pipe");
    let mut reader = io::BufReader::new(stdout);
    let mut line = String::new();
    let mut locked_seen = false;
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .expect("read child LOCKED handshake");
        if n == 0 {
            break;
        }
        if line.trim() == "LOCKED" {
            locked_seen = true;
            break;
        }
    }
    assert!(
        locked_seen,
        "child should report LOCKED before contention test; last line: {line:?}"
    );

    // FailFast must reject while child process holds kernel lock.
    let err = BundleLockGuard::acquire(bundle.run_dir(), LockAcquisitionPolicy::FailFast)
        .expect_err("parent must see BundleBusy while child holds kernel lock");
    assert!(matches!(err, BundleLockError::BundleBusy { .. }));

    // BoundedWait with short timeout must also fail closed while child holds lock.
    let err_wait = BundleLockGuard::acquire(
        bundle.run_dir(),
        LockAcquisitionPolicy::BoundedWait {
            timeout: Duration::from_millis(35),
            poll_interval: Duration::from_millis(5),
        },
    )
    .expect_err("short BoundedWait must fail closed when contender holds lock");
    assert!(matches!(err_wait, BundleLockError::BundleBusy { .. }));

    // Force-kill child process without running Drop -> kernel releases lock automatically.
    child.kill().expect("kill child lock holder");
    let _ = child.wait();
    assert!(
        lock_path.exists(),
        "persistent .bundle.lock remains after crash"
    );

    let recovered = BundleLockGuard::acquire(bundle.run_dir(), LockAcquisitionPolicy::FailFast)
        .expect("kernel must auto-release flock/LockFileEx on child crash");

    // Verify BoundedWait acquires cleanly once holder drops on another thread.
    let run_dir_owned = bundle.run_dir().to_path_buf();
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        drop(recovered);
    });
    let reacquired = BundleLockGuard::acquire(
        &run_dir_owned,
        LockAcquisitionPolicy::BoundedWait {
            timeout: Duration::from_millis(2_000),
            poll_interval: Duration::from_millis(5),
        },
    )
    .expect("BoundedWait should succeed once contender drops");
    drop(reacquired);
    releaser.join().unwrap();

    // Manifest finalization must continue to exclude persistent .bundle.lock.
    let manifest = bundle
        .finalize_signed_manifest(
            &swarm_torch_core::crypto::KeyPair::from_seed([0xB2; 32]).unwrap(),
        )
        .unwrap();
    assert!(
        !manifest
            .entries
            .iter()
            .any(|e| e.path.ends_with(BUNDLE_LOCK_FILE_NAME)),
        ".bundle.lock must never be hashed into manifest.json"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn st05_cas_write_once_collision_verifies_sha256_and_rejects_corrupt_target() {
    use sha2::{Digest, Sha256};

    let base = temp_dir("st05_cas_collision");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0xB3; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let payload = b"authentic-cas-payload-v1";
    let digest: [u8; 32] = Sha256::digest(payload).into();

    let path = write_once_cas_blob(bundle.run_dir(), &digest, payload).unwrap();
    // Idempotent duplicate write succeeds when on-disk SHA-256 matches.
    let path_again = write_once_cas_blob(bundle.run_dir(), &digest, payload).unwrap();
    assert_eq!(path, path_again);

    // Same-length bit-flip corruption in existing CAS blob must fail closed without overwriting.
    let mut corrupt_same_len = payload.to_vec();
    corrupt_same_len[0] ^= 0xFF;
    fs::write(&path, &corrupt_same_len).unwrap();
    let err_same = write_once_cas_blob(bundle.run_dir(), &digest, payload)
        .expect_err("corrupt existing CAS blob must fail SHA-256 verification");
    assert_eq!(err_same.kind(), io::ErrorKind::InvalidData);
    assert_eq!(
        fs::read(&path).unwrap(),
        corrupt_same_len,
        "write_once_cas_blob must never overwrite an existing corrupt CAS file"
    );

    // Length-mismatch corruption must also fail closed without overwriting.
    let corrupt_diff_len = b"truncated";
    fs::write(&path, corrupt_diff_len).unwrap();
    let err_diff = write_once_cas_blob(bundle.run_dir(), &digest, payload)
        .expect_err("size-mismatched CAS blob must fail closed");
    assert_eq!(err_diff.kind(), io::ErrorKind::InvalidData);
    assert_eq!(fs::read(&path).unwrap(), corrupt_diff_len);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn st05_ndjson_durability_policy_and_sink_write_graph_unlocked() {
    use swarm_torch_core::crypto::KeyPair;

    let base = temp_dir("st05_ndjson_and_sink_graph");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0xB4; 16]);
    let session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();

    // Writing graph through session.sink() while holding lock_guard must succeed without self-contention.
    let graph = GraphV1::default();
    session
        .sink()
        .write_graph(&graph)
        .expect("sink.write_graph must reuse held lock_guard without BundleBusy");

    let trace_id = TraceId::from_bytes([0x11; 16]);
    {
        let bundle = session.sink().bundle();
        bundle
            .append_metric_with_durability(
                &MetricRecord {
                    schema_version: 1,
                    ts_unix_nanos: 100,
                    trace_id,
                    span_id: None,
                    name: "loss".to_string(),
                    value: 0.25,
                    unit: None,
                    attrs: AttrMap::new(),
                },
                DurabilityPolicyV1::SyncEachAppend,
            )
            .unwrap();
        bundle
            .append_event(&EventRecord {
                schema_version: 1,
                ts_unix_nanos: 101,
                trace_id,
                span_id: None,
                name: "step".to_string(),
                attrs: AttrMap::new(),
            })
            .unwrap();
        bundle.sync_required_v1().unwrap();
    }

    let signer = KeyPair::from_seed([0xB5; 32]).unwrap();
    session.commit_and_close(Some(&signer)).unwrap();

    let loaded = RunArtifactBundle::open(&run_dir)
        .unwrap()
        .verify_and_load(&TrustPolicy::strict_with_key(*signer.public_key()))
        .expect("strict verification after durability sync & seal should pass");
    assert!(loaded.status().is_strict());

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn st15_session_cache_blocks_trust_laundering_invalidates_outputs_and_rejects_uppercase_hex() {
    let base = temp_dir("st15_session_trust_and_hex");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([0xC1; 16]);
    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();

    // Register an initial trusted input dataset:
    let ingest = make_source_node("ingest/raw");
    session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            SourceDescriptorV0 {
                uri: "http://trusted/raw".to_string(),
                content_type: "application/octet-stream".to_string(),
                auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                etag_or_version: None,
            },
            None,
            &ingest,
        )
        .unwrap();

    // Materialize a SandboxedExtension node -> output is Untrusted under sandbox profile:
    let mut node = make_transform_node(
        "transform/norm",
        &["dataset://ns/raw"],
        &["dataset://ns/clean"],
        ExecutionTrust::SandboxedExtension,
    );
    let mat_outputs = vec![OutputSpec {
        asset_key: "dataset://ns/clean".to_string(),
        schema: None,
        rows: Some(10),
        bytes: Some(100),
    }];
    let core_outputs = vec![OutputSpecCore {
        asset_key: "dataset://ns/clean".to_string(),
        schema: None,
    }];
    session
        .materialize_node_outputs(&node, &mat_outputs, 100, CacheDecisionV0::Miss, 1)
        .unwrap();
    assert_eq!(
        session.verify_cache_hit(&node, &core_outputs).unwrap(),
        CacheDecisionV0::Hit
    );

    // Switching execution_trust to Core OR UnsafeExtension MUST NOT return a cache hit!
    node.execution_trust = ExecutionTrust::Core;
    assert_eq!(
        session.verify_cache_hit(&node, &core_outputs).unwrap(),
        CacheDecisionV0::Miss
    );
    node.execution_trust = ExecutionTrust::UnsafeExtension;
    assert_eq!(
        session.verify_cache_hit(&node, &core_outputs).unwrap(),
        CacheDecisionV0::Miss
    );

    // Invalidating node outputs removes them cleanly and yields Miss:
    node.execution_trust = ExecutionTrust::SandboxedExtension;
    session.invalidate_node_outputs(&node).unwrap();
    assert_eq!(
        session.verify_cache_hit(&node, &core_outputs).unwrap(),
        CacheDecisionV0::Miss
    );

    // Uppercase hex must fail strict lowercase fingerprint parsing in CAS:
    assert!(
        super::cas::parse_cas_uri_digest(&format!("cas://sha256/{}", "A".repeat(64))).is_none()
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn st15_verify_committed_head_rejects_tampered_idle_lineage_snapshot() {
    use crate::coordinator::{
        RoundCoordinatorConfig, SwarmRoundCoordinator, UpdateKindV1, WeightingPolicyV1,
    };
    use swarm_torch_core::aggregation::RobustAggregation;
    use swarm_torch_models::simple::LinearModel;

    let base = temp_dir("st15_tampered_idle_lineage");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([0xC2; 16]);

    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();

    let model = LinearModel::new(1, 1).unwrap();
    let authorized_peer = swarm_torch_core::crypto::KeyPair::from_seed([0xC4; 32])
        .unwrap()
        .peer_id();
    let config = RoundCoordinatorConfig {
        update_kind: UpdateKindV1::ModelDeltaV1 {
            server_learning_rate: 1.0,
        },
        aggregation: RobustAggregation::FedAvg,
        weighting_policy: WeightingPolicyV1::EqualPeerVote,
        authorized_cohort: [authorized_peer].into_iter().collect(),
        min_quorum: 1,
        max_staleness: 0,
        max_clock_skew_secs: 60,
    };
    let _coord =
        SwarmRoundCoordinator::initialize(run_id, config, model, &mut session, 1_000).unwrap();
    drop(session);

    // Modify `datasets/lineage.json` on an idle committed bundle (no uncommitted NDJSON tail):
    let lineage_path = run_dir.join("datasets").join("lineage.json");
    let mut bytes = fs::read(&lineage_path).unwrap();
    bytes.push(b'\n');
    fs::write(&lineage_path, &bytes).unwrap();

    let bundle = RunArtifactBundle::open(&run_dir).unwrap();
    let err = bundle
        .verify_committed_head(&TrustPolicy::LegacyPermissive)
        .expect_err(
            "tampered datasets/lineage.json on idle bundle must fail verify_committed_head",
        );
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn st15_uppercase_hex_unbound_cas_reopened_cache_laundering_and_dp_head_fail_closed() {
    use super::cas::{
        read_committed_head_v1, CheckpointContentDigestV1, CheckpointRecipeKeyV1, CommittedHeadV1,
        COMMITTED_HEAD_SCHEMA_V1,
    };
    use super::dp_ledger::{
        parse_and_validate_dp_ledger_slice, pre_release_debit_dp_accountant,
        reconstruct_dp_accountant_from_committed_head, DP_LEDGER_REL_PATH,
    };
    use swarm_torch_core::dp::{
        DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1,
        PrivacyAccountingClassificationV1, RdpAccountantV1,
    };

    // 1. Canonical lowercase hex enforcement across CAS / committed_head / dp_ledger.
    let upper = "AA".repeat(32);
    assert!(CheckpointContentDigestV1::from_hex(&upper).is_err());
    assert!(CheckpointRecipeKeyV1::from_hex(&upper).is_err());

    let base = temp_dir("st15_adversarial_edge");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([0xC5; 16]);
    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();

    let ingest = make_source_node("ingest/raw");
    session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            SourceDescriptorV0 {
                uri: "http://trusted/raw".to_string(),
                content_type: "application/octet-stream".to_string(),
                auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                etag_or_version: None,
            },
            None,
            &ingest,
        )
        .unwrap();

    let mut node = make_transform_node(
        "transform/ext",
        &["dataset://ns/raw"],
        &["dataset://ns/ext_out"],
        ExecutionTrust::SandboxedExtension,
    );
    let mat_outputs = vec![OutputSpec {
        asset_key: "dataset://ns/ext_out".to_string(),
        schema: None,
        rows: Some(4),
        bytes: Some(32),
    }];
    let core_outputs = vec![OutputSpecCore {
        asset_key: "dataset://ns/ext_out".to_string(),
        schema: None,
    }];

    // 2. Atomic CAS binding validation rejects missing CAS blob without mutating registry!
    let mut bad_bindings = std::collections::BTreeMap::new();
    bad_bindings.insert(
        "dataset://ns/ext_out".to_string(),
        (
            format!("cas://sha256/{}", "ab".repeat(32)),
            "application/octet-stream".to_string(),
        ),
    );
    assert!(session
        .materialize_node_outputs_with_cas_bindings(
            &node,
            &mat_outputs,
            &bad_bindings,
            100,
            CacheDecisionV0::Miss,
            1,
        )
        .is_err());
    assert!(session.fingerprint("dataset://ns/ext_out").is_none());

    // Empty content_type binding also fails closed before mutation.
    let blob_payload = b"verified-output-bytes";
    let digest: [u8; 32] = {
        use sha2::Digest;
        sha2::Sha256::digest(blob_payload).into()
    };
    super::io::write_once_cas_blob(&run_dir, &digest, blob_payload).unwrap();
    bad_bindings.insert(
        "dataset://ns/ext_out".to_string(),
        (
            format!("cas://sha256/{}", hex_lower(&digest)),
            String::new(),
        ),
    );
    assert!(session
        .materialize_node_outputs_with_cas_bindings(
            &node,
            &mat_outputs,
            &bad_bindings,
            100,
            CacheDecisionV0::Miss,
            1,
        )
        .is_err());
    assert!(session.fingerprint("dataset://ns/ext_out").is_none());

    // Valid atomic CAS binding succeeds, and survives DataOpsSession reopen while still blocking
    // SandboxedExtension <-> UnsafeExtension <-> Core cache laundering!
    let mut good_bindings = std::collections::BTreeMap::new();
    good_bindings.insert(
        "dataset://ns/ext_out".to_string(),
        (
            format!("cas://sha256/{}", hex_lower(&digest)),
            "application/octet-stream".to_string(),
        ),
    );
    session
        .materialize_node_outputs_with_cas_bindings(
            &node,
            &mat_outputs,
            &good_bindings,
            100,
            CacheDecisionV0::Miss,
            1,
        )
        .unwrap();
    drop(session);

    let reopened = DataOpsSession::open(
        &run_dir,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    assert_eq!(
        reopened.verify_cache_hit(&node, &core_outputs).unwrap(),
        CacheDecisionV0::Hit
    );
    node.execution_trust = ExecutionTrust::UnsafeExtension;
    assert_eq!(
        reopened.verify_cache_hit(&node, &core_outputs).unwrap(),
        CacheDecisionV0::Miss
    );
    node.execution_trust = ExecutionTrust::Core;
    assert_eq!(
        reopened.verify_cache_hit(&node, &core_outputs).unwrap(),
        CacheDecisionV0::Miss
    );
    drop(reopened);

    // 3. committed_head.json rejects uppercase manifest_sha256.
    let bad_head = CommittedHeadV1 {
        schema_version: COMMITTED_HEAD_SCHEMA_V1,
        run_id,
        generation: 1,
        round_id: 1,
        manifest_sha256: "AA".repeat(32),
        resume_state_cas_uri: format!("cas://sha256/{}", "bb".repeat(32)),
        committed_unix_nanos: 1,
    };
    super::cas::write_committed_head_v1(&run_dir, &bad_head).unwrap();
    assert_eq!(
        read_committed_head_v1(&run_dir).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );

    // 4. reconstruct_dp_accountant_from_committed_head rejects unexpected tail when
    // after_cas_crash_tail_allowed == false and rejects uppercase hex in dp/ledger.ndjson.
    let dp_profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 10.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
    };
    let mut acc = RdpAccountantV1::new(dp_profile).unwrap();
    pre_release_debit_dp_accountant(
        &run_dir,
        run_id,
        1,
        &"00".repeat(32),
        &mut acc,
        1.0,
        0.5,
        10,
    )
    .unwrap();
    let bundle = RunArtifactBundle::open(&run_dir).unwrap();
    let manifest_r1 = bundle
        .finalize_signed_manifest(
            &swarm_torch_core::crypto::KeyPair::from_seed([0xC6; 32]).unwrap(),
        )
        .unwrap();
    pre_release_debit_dp_accountant(
        &run_dir,
        run_id,
        2,
        &"11".repeat(32),
        &mut acc,
        1.0,
        0.5,
        20,
    )
    .unwrap();
    assert!(
        reconstruct_dp_accountant_from_committed_head(&bundle, &manifest_r1, dp_profile, false)
            .is_err(),
        "uncommitted DP ledger tail must fail when after_cas_crash_tail_allowed == false"
    );
    let ledger_raw = fs::read_to_string(run_dir.join(DP_LEDGER_REL_PATH)).unwrap();
    let upper_ledger = ledger_raw.replace(&"00".repeat(32), &"0A".repeat(32));
    assert!(parse_and_validate_dp_ledger_slice(upper_ledger.as_bytes()).is_err());

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn st15_verify_committed_head_rejects_newline_tail_snapshot_tamper_symlink_manifest_and_dangling_symlink(
) {
    use crate::coordinator::{
        RoundCoordinatorConfig, SwarmRoundCoordinator, UpdateKindV1, WeightingPolicyV1,
    };
    use swarm_torch_core::aggregation::RobustAggregation;
    use swarm_torch_core::crypto::KeyPair;
    use swarm_torch_models::simple::LinearModel;

    let base = temp_dir("st15_head_tamper_and_symlink");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([0xC7; 16]);
    let signer = KeyPair::from_seed([0xC8; 32]).unwrap();
    let strict_policy = TrustPolicy::strict_with_key(*signer.public_key());

    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();
    let peer = KeyPair::from_seed([0xC9; 32]).unwrap().peer_id();
    let cfg = RoundCoordinatorConfig {
        update_kind: UpdateKindV1::ModelDeltaV1 {
            server_learning_rate: 1.0,
        },
        aggregation: RobustAggregation::FedAvg,
        weighting_policy: WeightingPolicyV1::EqualPeerVote,
        authorized_cohort: [peer].into_iter().collect(),
        min_quorum: 1,
        max_staleness: 0,
        max_clock_skew_secs: 60,
    };
    let _coord = SwarmRoundCoordinator::initialize(
        run_id,
        cfg,
        LinearModel::new(1, 1).unwrap(),
        &mut session,
        1_000,
    )
    .unwrap();
    drop(session);

    let bundle = RunArtifactBundle::open(&run_dir).unwrap();
    let signed_manifest = bundle.finalize_signed_manifest(&signer).unwrap();
    let manifest_sha = {
        use sha2::Digest;
        hex_lower(&sha2::Sha256::digest(
            serde_json::to_vec_pretty(&signed_manifest).unwrap(),
        ))
    };
    let mut head = super::cas::read_committed_head_v1(&run_dir).unwrap();
    head.manifest_sha256 = manifest_sha;
    super::cas::write_committed_head_v1(&run_dir, &head).unwrap();

    // 1. Appending `\n` to `datasets/materializations.ndjson` + tampering `datasets/lineage.json`
    //    MUST NOT activate crash-recovery snapshot exemption -> fails with InvalidData!
    let mat_path = run_dir.join("datasets").join("materializations.ndjson");
    let orig_mat = fs::read(&mat_path).unwrap();
    let mut nl_mat = orig_mat.clone();
    nl_mat.push(b'\n');
    fs::write(&mat_path, &nl_mat).unwrap();
    let lin_path = run_dir.join("datasets").join("lineage.json");
    let orig_lin = fs::read(&lin_path).unwrap();
    let mut bad_lin = orig_lin.clone();
    bad_lin.push(b'\n');
    fs::write(&lin_path, &bad_lin).unwrap();

    assert_eq!(
        bundle
            .verify_committed_head(&strict_policy)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
    fs::write(&mat_path, &orig_mat).unwrap();
    fs::write(&lin_path, &orig_lin).unwrap();

    // 2. Replacing `manifest.json` with a symlink MUST fail closed with InvalidData (not crash-recover!).
    let manifest_path = run_dir.join("manifest.json");
    let orig_manifest_bytes = fs::read(&manifest_path).unwrap();
    let outside_manifest = base.join("outside_manifest.json");
    fs::write(&outside_manifest, &orig_manifest_bytes).unwrap();
    fs::remove_file(&manifest_path).unwrap();
    #[cfg(unix)]
    let sym_ok = std::os::unix::fs::symlink(&outside_manifest, &manifest_path).is_ok();
    #[cfg(windows)]
    let sym_ok = std::os::windows::fs::symlink_file(&outside_manifest, &manifest_path).is_ok();
    if sym_ok {
        assert_eq!(
            bundle
                .verify_committed_head(&strict_policy)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let _ = fs::remove_file(&manifest_path);
    }
    fs::write(&manifest_path, &orig_manifest_bytes).unwrap();

    // 3. Replacing a manifest-listed file with a dangling symlink returns InvalidData, not NotFound!
    let dangling_target = base.join("does_not_exist_target.bin");
    let link_path = run_dir.join("datasets").join("escape_dangling");
    #[cfg(unix)]
    let d_ok = std::os::unix::fs::symlink(&dangling_target, &link_path).is_ok();
    #[cfg(windows)]
    let d_ok = std::os::windows::fs::symlink_file(&dangling_target, &link_path).is_ok();
    if d_ok {
        let mut m_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        m_json["signature"] = serde_json::Value::Null;
        let mut entries_arr = m_json["entries"].as_array().cloned().unwrap();
        let mut extra = entries_arr[0].clone();
        extra["path"] = serde_json::Value::String("datasets/escape_dangling".to_string());
        extra["required"] = serde_json::Value::Bool(false);
        entries_arr.push(extra);
        m_json["entries"] = serde_json::Value::Array(entries_arr);
        fs::write(&manifest_path, serde_json::to_vec_pretty(&m_json).unwrap()).unwrap();
        let err = bundle.validate_manifest().unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidData,
            "dangling symlink should return InvalidData, got {err:?}"
        );
    }

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn st15_bundle_create_rejects_dangling_symlink_and_verify_head_rejects_oversized_or_symlinked_archive(
) {
    use crate::coordinator::{
        RoundCoordinatorConfig, SwarmRoundCoordinator, UpdateKindV1, WeightingPolicyV1,
    };
    use swarm_torch_core::aggregation::RobustAggregation;
    use swarm_torch_core::crypto::KeyPair;
    use swarm_torch_models::simple::LinearModel;

    let base = temp_dir("st15_dangling_create_and_oversized_manifest");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(base.join("runs")).unwrap();

    let run_id = RunId::from_bytes([0xC9; 16]);
    let symlink_run_dir = base.join("runs").join(run_id.to_string());
    let missing_target = base.join("missing_bundle_target");

    #[cfg(unix)]
    let link_ok = std::os::unix::fs::symlink(&missing_target, &symlink_run_dir).is_ok();
    #[cfg(windows)]
    let link_ok = std::os::windows::fs::symlink_dir(&missing_target, &symlink_run_dir).is_ok();
    if link_ok {
        let err = RunArtifactBundle::create(&base, run_id).unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidData,
            "RunArtifactBundle::create must fail closed on dangling symlink, got {err:?}"
        );
        #[cfg(unix)]
        let _ = fs::remove_file(&symlink_run_dir);
        #[cfg(windows)]
        let _ = fs::remove_dir(&symlink_run_dir);
    }

    // Now test oversized manifest.json and symlinked archived manifest in verify_committed_head.
    let head_base = base.join("head_test");
    fs::create_dir_all(&head_base).unwrap();
    let signer = KeyPair::from_seed([0xCA; 32]).unwrap();
    let strict_policy = TrustPolicy::strict_with_key(*signer.public_key());
    let mut session = DataOpsSession::create(
        &head_base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();
    let peer = KeyPair::from_seed([0xCB; 32]).unwrap().peer_id();
    let cfg = RoundCoordinatorConfig {
        update_kind: UpdateKindV1::ModelDeltaV1 {
            server_learning_rate: 1.0,
        },
        aggregation: RobustAggregation::FedAvg,
        weighting_policy: WeightingPolicyV1::EqualPeerVote,
        authorized_cohort: [peer].into_iter().collect(),
        min_quorum: 1,
        max_staleness: 0,
        max_clock_skew_secs: 60,
    };
    let _coord = SwarmRoundCoordinator::initialize(
        run_id,
        cfg,
        LinearModel::new(1, 1).unwrap(),
        &mut session,
        1_000,
    )
    .unwrap();
    drop(session);

    let bundle = RunArtifactBundle::open(&run_dir).unwrap();
    let signed_manifest = bundle.finalize_signed_manifest(&signer).unwrap();
    let signed_bytes = serde_json::to_vec_pretty(&signed_manifest).unwrap();
    let signed_sha = {
        use sha2::Digest;
        hex_lower(&sha2::Sha256::digest(&signed_bytes))
    };
    let mut head = super::cas::read_committed_head_v1(&run_dir).unwrap();
    head.manifest_sha256 = signed_sha;
    super::cas::write_committed_head_v1(&run_dir, &head).unwrap();
    super::cas::write_archived_generation_manifest_bytes(&run_dir, head.generation, &signed_bytes)
        .unwrap();

    // 1. Oversized manifest.json (> MAX_MANIFEST_BYTES) MUST fail closed instead of unbounded streaming.
    let manifest_path = run_dir.join("manifest.json");
    let oversized = vec![b' '; (MAX_MANIFEST_BYTES as usize) + 1];
    fs::write(&manifest_path, &oversized).unwrap();
    let oversized_err = bundle.verify_committed_head(&strict_policy).unwrap_err();
    assert_eq!(oversized_err.kind(), io::ErrorKind::InvalidData);

    // 2. Missing manifest.json + symlinked archived generation manifest MUST fail closed with InvalidData.
    fs::remove_file(&manifest_path).unwrap();
    let archive_rel = super::cas::committed_manifest_archive_rel_path(head.generation);
    let archive_path = run_dir.join(&archive_rel);
    let outside_archive = base.join("outside_archive.json");
    fs::write(&outside_archive, &signed_bytes).unwrap();
    fs::remove_file(&archive_path).unwrap();
    #[cfg(unix)]
    let arch_sym_ok = std::os::unix::fs::symlink(&outside_archive, &archive_path).is_ok();
    #[cfg(windows)]
    let arch_sym_ok = std::os::windows::fs::symlink_file(&outside_archive, &archive_path).is_ok();
    if arch_sym_ok {
        let arch_err = bundle.verify_committed_head(&strict_policy).unwrap_err();
        assert_eq!(arch_err.kind(), io::ErrorKind::InvalidData);
    }

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn register_source_and_cas_bindings_reject_oversized_content_type_and_custom_auth_atomically() {
    use swarm_torch_core::dataops::{
        AuthModeMarker, MAX_AUTH_MODE_CUSTOM_LEN, MAX_SOURCE_CONTENT_TYPE_LEN,
    };

    let base = temp_dir("session_source_bounds_parity");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let run_id = RunId::from_bytes([0xCE; 16]);
    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();
    let ingest = make_source_node("ingest/raw");

    let bad_ct_src = SourceDescriptorV0 {
        uri: "s3://bucket/raw.parquet".to_string(),
        content_type: "c".repeat(MAX_SOURCE_CONTENT_TYPE_LEN + 1),
        auth_mode: AuthModeMarker::None,
        etag_or_version: None,
    };
    let ct_err = session
        .register_source(
            "dataset://ns/bad_ct",
            TrustClass::Trusted,
            bad_ct_src,
            None,
            &ingest,
        )
        .expect_err("oversized content_type must fail");
    assert_eq!(ct_err.kind(), io::ErrorKind::InvalidInput);
    assert!(session.fingerprint("dataset://ns/bad_ct").is_none());

    let bad_auth_src = SourceDescriptorV0 {
        uri: "s3://bucket/raw.parquet".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: AuthModeMarker::Custom("m".repeat(MAX_AUTH_MODE_CUSTOM_LEN + 1)),
        etag_or_version: None,
    };
    let auth_err = session
        .register_source(
            "dataset://ns/bad_auth",
            TrustClass::Trusted,
            bad_auth_src,
            None,
            &ingest,
        )
        .expect_err("oversized custom auth_mode must fail");
    assert_eq!(auth_err.kind(), io::ErrorKind::InvalidInput);
    assert!(session.fingerprint("dataset://ns/bad_auth").is_none());

    session
        .register_source(
            "dataset://ns/raw",
            TrustClass::Trusted,
            SourceDescriptorV0 {
                uri: " s3://user:secret@bucket/raw.parquet?token=1#frag ".to_string(),
                content_type: " Application/Parquet ".to_string(),
                auth_mode: AuthModeMarker::BearerToken,
                etag_or_version: Some(" v1 ".to_string()),
            },
            None,
            &ingest,
        )
        .unwrap();
    assert_eq!(
        session.asset_content_type("dataset://ns/raw"),
        Some("application/parquet")
    );

    let blob_payload = b"cas-payload";
    let digest: [u8; 32] = {
        use sha2::Digest;
        sha2::Sha256::digest(blob_payload).into()
    };
    super::io::write_once_cas_blob(&run_dir, &digest, blob_payload).unwrap();
    let cas_uri = format!("cas://sha256/{}", hex_lower(&digest));

    let node = make_transform_node(
        "transform/core",
        &["dataset://ns/raw"],
        &["dataset://ns/out"],
        ExecutionTrust::Core,
    );
    let mat_outputs = vec![OutputSpec {
        asset_key: "dataset://ns/out".to_string(),
        schema: None,
        rows: Some(1),
        bytes: Some(11),
    }];

    // Whitespace-only or oversized content_type in cas_bindings must fail with InvalidInput without mutating state.
    for bad_ct in [
        "x".repeat(MAX_SOURCE_CONTENT_TYPE_LEN + 1),
        String::new(),
        "   \t ".to_string(),
    ] {
        let mut bad_bindings = std::collections::BTreeMap::new();
        bad_bindings.insert("dataset://ns/out".to_string(), (cas_uri.clone(), bad_ct));
        let bind_err = session
            .materialize_node_outputs_with_cas_bindings(
                &node,
                &mat_outputs,
                &bad_bindings,
                100,
                CacheDecisionV0::Miss,
                1,
            )
            .expect_err("invalid cas_bindings content_type must fail");
        assert_eq!(bind_err.kind(), io::ErrorKind::InvalidInput);
        assert!(session.fingerprint("dataset://ns/out").is_none());
    }

    // Valid mixed-case/untrimmed content_type in cas_bindings is sanitized consistently.
    let mut good_bindings = std::collections::BTreeMap::new();
    good_bindings.insert(
        "dataset://ns/out".to_string(),
        (cas_uri.clone(), " Application/Octet-Stream ".to_string()),
    );
    session
        .materialize_node_outputs_with_cas_bindings(
            &node,
            &mat_outputs,
            &good_bindings,
            100,
            CacheDecisionV0::Miss,
            1,
        )
        .unwrap();
    assert_eq!(
        session.asset_content_type("dataset://ns/out"),
        Some("application/octet-stream")
    );

    // Sibling `bind_output_cas_uri` enforces identical bounds + sanitization and leaves prior source intact on rejection.
    for bad_ct in [
        "y".repeat(MAX_SOURCE_CONTENT_TYPE_LEN + 1),
        String::new(),
        "   ".to_string(),
    ] {
        let err = session
            .bind_output_cas_uri("dataset://ns/out", &cas_uri, &bad_ct)
            .expect_err("bind_output_cas_uri must reject empty/oversized content_type");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            session.asset_content_type("dataset://ns/out"),
            Some("application/octet-stream")
        );
    }
    session
        .bind_output_cas_uri("dataset://ns/out", &cas_uri, " Text/CSV ")
        .unwrap();
    assert_eq!(
        session.asset_content_type("dataset://ns/out"),
        Some("text/csv")
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn wave16b2_certified_dp_ledger_enforces_sealed_proof_binding_classification_segregation_and_crash_tail(
) {
    use super::dp_ledger::{
        append_dp_release_record_durable, parse_and_validate_dp_ledger_slice,
        pre_release_debit_dp_accountant, pre_release_debit_verified_dp_accountant,
        read_dp_ledger_entries, reconstruct_dp_accountant_from_committed_head,
        reconstruct_dp_accountant_from_verified_bundle, CryptoErrorOrIo, DP_LEDGER_REL_PATH,
        EXPERIMENTAL_DP_CLASSIFICATION_V1, VERIFIED_DP_CLASSIFICATION_V1,
    };
    use swarm_torch_core::crypto::{CryptoError, KeyPair};
    use swarm_torch_core::dp::{
        DpAdjacencyV1, DpConversionFormulaV1, DpFedAvgMechanismV1, DpFedAvgRoundInputV1,
        DpMechanismProfileV1, DpNoiseEntropyV1, DpUnitV1, PrivacyAccountingClassificationV1,
        RdpAccountantV1, SampledDpCohortV1,
    };
    use swarm_torch_core::traits::PeerId;

    let base = temp_dir("wave16b2_certified_dp_ledger");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0xE4; 16]);
    let other_run_id = RunId::from_bytes([0xE5; 16]);
    let parent_bytes = [0x1Au8; 32];
    let parent_hex = hex_lower(&parent_bytes);
    let other_parent_hex = hex_lower(&[0x3Cu8; 32]);
    let peers: Vec<PeerId> = (1u8..=4).map(|i| PeerId::new([i; 32])).collect();

    let cert_profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 10.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    let exp_profile = DpMechanismProfileV1 {
        classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
        ..cert_profile
    };

    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    let run_dir = bundle.run_dir().to_path_buf();
    let signer = KeyPair::from_seed([0x91; 32]).unwrap();

    let mut cert_acc = RdpAccountantV1::new_verified_mechanism(cert_profile).unwrap();

    // 1. Legacy unsealed pre_release_debit_dp_accountant MUST fail closed on a certified accountant.
    assert!(matches!(
        pre_release_debit_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex,
            &mut cert_acc,
            1.0,
            0.5,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    assert_eq!(cert_acc.committed_rounds(), 0);
    assert!(read_dp_ledger_entries(&run_dir).unwrap().is_empty());

    // 2. Explicit non-production test seed proof (both or either sampling/noise explicit) MUST be rejected
    //    by pre_release_debit_verified_dp_accountant.
    let test_cohort = SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(
        &peers,
        0.5,
        [0x44; 32],
        *run_id.as_bytes(),
        1,
    )
    .unwrap();
    let test_outcome = DpFedAvgMechanismV1::execute_round_v1(
        cert_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: *run_id.as_bytes(),
            round_id: 1,
            parent_digest_bytes: parent_bytes,
            parameter_dim: 4,
            sampled_cohort: &test_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex,
            &mut cert_acc,
            &test_outcome.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    let explicit_cohort_os_noise = DpFedAvgMechanismV1::execute_round_v1(
        cert_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: *run_id.as_bytes(),
            round_id: 1,
            parent_digest_bytes: parent_bytes,
            parameter_dim: 4,
            sampled_cohort: &test_cohort,
            noise_entropy: DpNoiseEntropyV1::generate_os_csprng().unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(!explicit_cohort_os_noise
        .proof
        .is_production_certified_entropy());
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex,
            &mut cert_acc,
            &explicit_cohort_os_noise.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    assert_eq!(cert_acc.committed_rounds(), 0);
    assert!(read_dp_ledger_entries(&run_dir).unwrap().is_empty());

    // 3. Genuine OS CSPRNG proof binds exact (run_id, round_id, parent_digest, profile) and rejects mismatches!
    let os_cohort_r1 =
        SampledDpCohortV1::sample_poisson_os_rng_v1(&peers, 0.5, *run_id.as_bytes(), 1).unwrap();
    let os_cohort_explicit_noise = DpFedAvgMechanismV1::execute_round_v1(
        cert_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: *run_id.as_bytes(),
            round_id: 1,
            parent_digest_bytes: parent_bytes,
            parameter_dim: 4,
            sampled_cohort: &os_cohort_r1,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x56; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(!os_cohort_explicit_noise
        .proof
        .is_production_certified_entropy());
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex,
            &mut cert_acc,
            &os_cohort_explicit_noise.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));

    let os_outcome_r1 = DpFedAvgMechanismV1::execute_round_v1(
        cert_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: *run_id.as_bytes(),
            round_id: 1,
            parent_digest_bytes: parent_bytes,
            parameter_dim: 4,
            sampled_cohort: &os_cohort_r1,
            noise_entropy: DpNoiseEntropyV1::generate_os_csprng().unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(os_outcome_r1.proof.is_production_certified_entropy());

    // Passing an experimental accountant to pre_release_debit_verified_dp_accountant MUST fail closed.
    let mut exp_acc = RdpAccountantV1::new(exp_profile).unwrap();
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex,
            &mut exp_acc,
            &os_outcome_r1.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    assert_eq!(exp_acc.committed_rounds(), 0);

    // Budget exhaustion in pre_release_debit_verified_dp_accountant MUST fail closed BEFORE writing disk or mutating accountant.
    let tiny_budget_profile = DpMechanismProfileV1 {
        max_epsilon: 0.01,
        ..cert_profile
    };
    let mut tiny_budget_acc = RdpAccountantV1::new_verified_mechanism(tiny_budget_profile).unwrap();
    let tiny_budget_outcome = DpFedAvgMechanismV1::execute_round_v1(
        tiny_budget_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: *run_id.as_bytes(),
            round_id: 1,
            parent_digest_bytes: parent_bytes,
            parameter_dim: 4,
            sampled_cohort: &os_cohort_r1,
            noise_entropy: DpNoiseEntropyV1::generate_os_csprng().unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex,
            &mut tiny_budget_acc,
            &tiny_budget_outcome.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(CryptoError::PrivacyBudgetExhausted))
    ));
    assert_eq!(tiny_budget_acc.committed_rounds(), 0);
    assert!(read_dp_ledger_entries(&run_dir).unwrap().is_empty());

    // Wrong run_id, wrong round_id, wrong parent_digest, uppercase hex, wrong profile all fail closed.
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            other_run_id,
            1,
            &parent_hex,
            &mut cert_acc,
            &os_outcome_r1.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            2,
            &parent_hex,
            &mut cert_acc,
            &os_outcome_r1.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &other_parent_hex,
            &mut cert_acc,
            &os_outcome_r1.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex.to_uppercase(),
            &mut cert_acc,
            &os_outcome_r1.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Io(_))
    ));
    let mut wrong_profile_acc = RdpAccountantV1::new_verified_mechanism(DpMechanismProfileV1 {
        noise_multiplier: 2.0,
        ..cert_profile
    })
    .unwrap();
    assert!(matches!(
        pre_release_debit_verified_dp_accountant(
            &run_dir,
            run_id,
            1,
            &parent_hex,
            &mut wrong_profile_acc,
            &os_outcome_r1.proof,
            1_000,
        ),
        Err(CryptoErrorOrIo::Crypto(
            CryptoError::InvalidPrivacyParameter
        ))
    ));
    assert_eq!(cert_acc.committed_rounds(), 0);

    // 4. Valid certified OS CSPRNG debit succeeds and records guarantees_differential_privacy = true.
    //    Before round-1 debit, finalize an initial empty manifest (`manifest_r0`, without `dp/ledger.ndjson`)
    //    to also verify the `None` branch of `reconstruct_dp_accountant_from_committed_head`.
    let manifest_r0 = bundle.finalize_signed_manifest(&signer).unwrap();
    let empty_restored =
        reconstruct_dp_accountant_from_committed_head(&bundle, &manifest_r0, cert_profile, false)
            .unwrap();
    assert!(empty_restored.guarantees_differential_privacy());
    assert_eq!(empty_restored.committed_rounds(), 0);

    let (newly_appended, entry1) = pre_release_debit_verified_dp_accountant(
        &run_dir,
        run_id,
        1,
        &parent_hex,
        &mut cert_acc,
        &os_outcome_r1.proof,
        1_000,
    )
    .unwrap();
    assert!(newly_appended);
    assert!(entry1.guarantees_differential_privacy);
    assert_eq!(
        entry1.accounting_classification,
        VERIFIED_DP_CLASSIFICATION_V1
    );
    assert_eq!(entry1.committed_rounds, 1);
    assert_eq!(cert_acc.committed_rounds(), 1);

    // Against `manifest_r0` (which does not list `dp/ledger.ndjson`), `after_cas_crash_tail_allowed = false`
    // MUST fail closed, while `after_cas_crash_tail_allowed = true` recovers the round-1 crash-tail debit.
    assert!(reconstruct_dp_accountant_from_committed_head(
        &bundle,
        &manifest_r0,
        cert_profile,
        false,
    )
    .is_err());
    let recovered_r1_from_r0 =
        reconstruct_dp_accountant_from_committed_head(&bundle, &manifest_r0, cert_profile, true)
            .unwrap();
    assert!(recovered_r1_from_r0.guarantees_differential_privacy());
    assert_eq!(recovered_r1_from_r0.committed_rounds(), 1);

    // Idempotent retry on duplicate `receipt_id` returns `Ok(false)` when all fields agree,
    // and fails closed on any conflicting payload field.
    assert!(!append_dp_release_record_durable(&run_dir, &entry1).unwrap());
    let mut conflicting_q = entry1.clone();
    conflicting_q.subsampling_rate_q = 0.75;
    assert!(append_dp_release_record_durable(&run_dir, &conflicting_q).is_err());
    let mut conflicting_parent = entry1.clone();
    conflicting_parent.parent_digest_hex = other_parent_hex.clone();
    assert!(append_dp_release_record_durable(&run_dir, &conflicting_parent).is_err());

    // Seal signed manifest covering round 1, then verify strict bundle reconstruction & classification segregation.
    let manifest_r1 = bundle.finalize_signed_manifest(&signer).unwrap();
    let verified_r1 = bundle
        .verify_and_load(&TrustPolicy::strict_with_key(*signer.public_key()))
        .unwrap();
    let restored_v =
        reconstruct_dp_accountant_from_verified_bundle(&verified_r1, cert_profile).unwrap();
    assert!(restored_v.guarantees_differential_privacy());
    assert_eq!(restored_v.committed_rounds(), 1);

    // Attempting to reconstruct a certified ledger as experimental MUST fail closed!
    assert!(matches!(
        reconstruct_dp_accountant_from_verified_bundle(&verified_r1, exp_profile),
        Err(CryptoErrorOrIo::Io(_))
    ));

    // 5. Simulate round 2 crash tail (debited to disk after manifest_r1, before new manifest),
    //    using DeterministicFullCohort (q = 1.0) + OS CSPRNG noise to also verify full-cohort certified provenance.
    let full_cohort_r2 = SampledDpCohortV1::full_cohort_v1(&peers, *run_id.as_bytes(), 2).unwrap();
    let os_outcome_r2 = DpFedAvgMechanismV1::execute_round_v1(
        cert_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: *run_id.as_bytes(),
            round_id: 2,
            parent_digest_bytes: parent_bytes,
            parameter_dim: 4,
            sampled_cohort: &full_cohort_r2,
            noise_entropy: DpNoiseEntropyV1::generate_os_csprng().unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(os_outcome_r2.proof.is_production_certified_entropy());
    let (newly_appended2, entry2) = pre_release_debit_verified_dp_accountant(
        &run_dir,
        run_id,
        2,
        &parent_hex,
        &mut cert_acc,
        &os_outcome_r2.proof,
        2_000,
    )
    .unwrap();
    assert!(newly_appended2);
    assert_eq!(entry2.committed_rounds, 2);

    // Strict committed-head read with crash tail forbidden fails closed; with crash tail allowed preserves round 2 debit!
    assert!(reconstruct_dp_accountant_from_committed_head(
        &bundle,
        &manifest_r1,
        cert_profile,
        false,
    )
    .is_err());
    let recovered_after_crash =
        reconstruct_dp_accountant_from_committed_head(&bundle, &manifest_r1, cert_profile, true)
            .unwrap();
    assert!(recovered_after_crash.guarantees_differential_privacy());
    assert_eq!(recovered_after_crash.committed_rounds(), 2);

    // 6. Direct append of a mixed-classification experimental entry onto a certified ledger MUST fail closed!
    let mut mixed_exp = entry2.clone();
    mixed_exp.receipt_id = hex_lower(&[0xAB; 32]);
    mixed_exp.committed_rounds = 3;
    mixed_exp.guarantees_differential_privacy = false;
    mixed_exp.accounting_classification = EXPERIMENTAL_DP_CLASSIFICATION_V1.to_string();
    assert!(append_dp_release_record_durable(&run_dir, &mixed_exp).is_err());

    // 7. Parse validation rejects tampered epsilon/alpha, mismatched bool/string pairs, ExampleLevel verified,
    //    out-of-range sigma_z noise_multiplier, non-monotonic committed_rounds, decreasing RDP orders,
    //    mutated clipping_radius_tau, and mixed-classification NDJSON slices.
    let mut bad_bool_str = entry1.clone();
    bad_bool_str.accounting_classification = EXPERIMENTAL_DP_CLASSIFICATION_V1.to_string();
    let bad_bytes = format!("{}\n", serde_json::to_string(&bad_bool_str).unwrap());
    assert!(parse_and_validate_dp_ledger_slice(bad_bytes.as_bytes()).is_err());

    let mut bad_false_verified_str = entry1.clone();
    bad_false_verified_str.guarantees_differential_privacy = false;
    let bad_false_verified_bytes = format!(
        "{}\n",
        serde_json::to_string(&bad_false_verified_str).unwrap()
    );
    assert!(parse_and_validate_dp_ledger_slice(bad_false_verified_bytes.as_bytes()).is_err());

    let mut bad_example_unit = entry1.clone();
    bad_example_unit.unit = DpUnitV1::ExampleLevel;
    let bad_unit_bytes = format!("{}\n", serde_json::to_string(&bad_example_unit).unwrap());
    assert!(parse_and_validate_dp_ledger_slice(bad_unit_bytes.as_bytes()).is_err());

    let mut bad_huge_sigma = entry1.clone();
    bad_huge_sigma.noise_multiplier = 300.0;
    let bad_sigma_bytes = format!("{}\n", serde_json::to_string(&bad_huge_sigma).unwrap());
    assert!(parse_and_validate_dp_ledger_slice(bad_sigma_bytes.as_bytes()).is_err());

    let mut bad_eps = entry1.clone();
    bad_eps.converted_epsilon = 0.0001;
    let bad_eps_bytes = format!("{}\n", serde_json::to_string(&bad_eps).unwrap());
    assert!(parse_and_validate_dp_ledger_slice(bad_eps_bytes.as_bytes()).is_err());

    let mut bad_alpha = entry1.clone();
    bad_alpha.optimal_alpha += 1.0;
    let bad_alpha_bytes = format!("{}\n", serde_json::to_string(&bad_alpha).unwrap());
    assert!(parse_and_validate_dp_ledger_slice(bad_alpha_bytes.as_bytes()).is_err());

    let mut bad_tau_drift = entry2.clone();
    bad_tau_drift.clipping_radius_tau = 2.0;
    let drift_bytes = format!(
        "{}\n{}\n",
        serde_json::to_string(&entry1).unwrap(),
        serde_json::to_string(&bad_tau_drift).unwrap()
    );
    assert!(parse_and_validate_dp_ledger_slice(drift_bytes.as_bytes()).is_err());

    let first_round_not_one = format!("{}\n", serde_json::to_string(&entry2).unwrap());
    assert!(parse_and_validate_dp_ledger_slice(first_round_not_one.as_bytes()).is_err());

    let mut bad_round_gap = entry2.clone();
    bad_round_gap.committed_rounds = 3;
    let gap_bytes = format!(
        "{}\n{}\n",
        serde_json::to_string(&entry1).unwrap(),
        serde_json::to_string(&bad_round_gap).unwrap()
    );
    assert!(parse_and_validate_dp_ledger_slice(gap_bytes.as_bytes()).is_err());

    let mut bad_decreasing_orders = entry1.clone();
    bad_decreasing_orders.receipt_id = hex_lower(&[0xCD; 32]);
    bad_decreasing_orders.committed_rounds = 3;
    let decreasing_bytes = format!(
        "{}\n{}\n{}\n",
        serde_json::to_string(&entry1).unwrap(),
        serde_json::to_string(&entry2).unwrap(),
        serde_json::to_string(&bad_decreasing_orders).unwrap()
    );
    assert!(parse_and_validate_dp_ledger_slice(decreasing_bytes.as_bytes()).is_err());

    let mixed_slice = format!(
        "{}\n{}\n",
        serde_json::to_string(&entry1).unwrap(),
        serde_json::to_string(&mixed_exp).unwrap()
    );
    assert!(parse_and_validate_dp_ledger_slice(mixed_slice.as_bytes()).is_err());

    // Verify ledger file on disk contains no participant_count / clipped_count / sampled_cohort leakage fields.
    let raw_ledger = fs::read_to_string(run_dir.join(DP_LEDGER_REL_PATH)).unwrap();
    assert!(!raw_ledger.contains("participant_count"));
    assert!(!raw_ledger.contains("clipped_count"));
    assert!(!raw_ledger.contains("sampled_cohort"));

    let _ = fs::remove_dir_all(&base);
}
