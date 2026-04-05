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
