//! Executable Draft 2020-12 JSON Schema conformance tests for `schemas/v1/*.schema.json`
//! (`ADR-0016`, `ADR-0023` §3.3, `P2-20a`, `W10-02`).

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use swarm_torch::artifacts::schema_check::{
    parse_and_check_schema_meta, validate_json_bytes_against_schema,
    validate_ndjson_bytes_against_schema, validate_verified_bundle_schemas, EVENT_SCHEMA_JSON,
    GRAPH_SCHEMA_JSON, LINEAGE_SCHEMA_JSON, MANIFEST_SCHEMA_JSON, MATERIALIZATION_SCHEMA_JSON,
    METRIC_SCHEMA_JSON, REGISTRY_SCHEMA_JSON, RUN_SCHEMA_JSON, SPAN_SCHEMA_JSON,
};
use swarm_torch::artifacts::{RunArtifactBundle, TrustPolicy};
use swarm_torch_core::crypto::KeyPair;
use swarm_torch_core::dataops::{
    AuthModeMarker, CacheDecisionV0, DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1,
    LineageEdgeV1, MaterializationRecordV1, MaterializationRecordV2, MaterializationStatusV0,
    SourceDescriptorV0, TransformAuditV0, TrustClass, UnsafeReasonV0, MATERIALIZATION_SCHEMA_V2,
};
use swarm_torch_core::observe::{
    AttrMap, AttrValue, EventRecord, MetricRecord, RunId, SpanId, SpanRecord, TraceId,
};
use swarm_torch_core::run_graph::{
    node_id_from_key, AssetRefV1, CanonParams, EdgeV1, ExecutionTrust, GraphV1, NodeV1, OpKind,
};

fn temp_dir(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "swarmtorch_schema_conformance_{prefix}_{pid}_{nanos}"
    ));
    dir
}

#[test]
fn all_nine_v1_schemas_have_valid_draft_2020_12_metadata() {
    let schemas = [
        ("manifest.schema.json", MANIFEST_SCHEMA_JSON),
        ("run.schema.json", RUN_SCHEMA_JSON),
        ("graph.schema.json", GRAPH_SCHEMA_JSON),
        ("registry.schema.json", REGISTRY_SCHEMA_JSON),
        ("lineage.schema.json", LINEAGE_SCHEMA_JSON),
        ("span.schema.json", SPAN_SCHEMA_JSON),
        ("event.schema.json", EVENT_SCHEMA_JSON),
        ("metric.schema.json", METRIC_SCHEMA_JSON),
        ("materialization.schema.json", MATERIALIZATION_SCHEMA_JSON),
    ];

    for (name, raw) in schemas {
        let parsed = parse_and_check_schema_meta(raw)
            .unwrap_or_else(|e| panic!("schema {name} failed meta check: {e}"));
        let id = parsed["$id"].as_str().unwrap();
        assert!(
            id.ends_with(name),
            "schema {name} $id {id} must end with filename"
        );
    }
}

#[test]
fn signed_bundle_with_u64_max_timestamps_conforms_to_all_v1_schemas() {
    let base = temp_dir("signed_bundle_u64_max");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0x42; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

    let node_a_id = node_id_from_key("ingest/raw");
    let node_b_id = node_id_from_key("transform/clean");
    let graph = GraphV1 {
        schema_version: 1,
        graph_id: Some("wave10-conformance-graph".to_string()),
        nodes: vec![
            NodeV1 {
                node_key: "ingest/raw".to_string(),
                node_id: Some(node_a_id),
                op_kind: OpKind::Data,
                op_type: "ingest".to_string(),
                inputs: vec![],
                outputs: vec![AssetRefV1 {
                    asset_key: "dataset://ns/raw".to_string(),
                    fingerprint: Some("a".repeat(64)),
                }],
                params: CanonParams::new(),
                code_ref: Some("swarm-torch@0.1.0".to_string()),
                unsafe_surface: false,
                execution_trust: ExecutionTrust::Core,
                node_def_hash: None,
                execution_hint: None,
                cache_policy: None,
                materialization_policy: None,
                resources: None,
                op_hash: None,
            },
            NodeV1 {
                node_key: "transform/clean".to_string(),
                node_id: Some(node_b_id),
                op_kind: OpKind::Data,
                op_type: "transform".to_string(),
                inputs: vec![AssetRefV1 {
                    asset_key: "dataset://ns/raw".to_string(),
                    fingerprint: Some("a".repeat(64)),
                }],
                outputs: vec![AssetRefV1 {
                    asset_key: "dataset://ns/clean".to_string(),
                    fingerprint: Some("b".repeat(64)),
                }],
                params: CanonParams::new(),
                code_ref: Some("swarm-torch@0.1.0".to_string()),
                unsafe_surface: false,
                execution_trust: ExecutionTrust::Core,
                node_def_hash: None,
                execution_hint: None,
                cache_policy: None,
                materialization_policy: None,
                resources: None,
                op_hash: None,
            },
        ],
        edges: vec![EdgeV1 {
            from_node_id: node_a_id,
            to_node_id: node_b_id,
            asset_key: Some("dataset://ns/raw".to_string()),
        }],
    };
    bundle.write_graph(&graph).unwrap();

    let entry = DatasetEntryV1 {
        asset_key: "dataset://ns/raw".to_string(),
        fingerprint_v0: "a".repeat(64),
        source_fingerprint_v0: "c".repeat(64),
        schema_hash_v0: "d".repeat(64),
        recipe_hash_v0: "0".repeat(64),
        trust: TrustClass::Trusted,
        source: Some(SourceDescriptorV0 {
            uri: "s3://bucket/raw.parquet".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: Some("v1".to_string()),
        }),
        schema: None,
        license_flags: vec!["Apache-2.0".to_string()],
        pii_tags: vec![],
    };
    bundle
        .write_dataset_registry(&DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![entry.clone()],
        })
        .unwrap();
    bundle.append_registry_update(&entry).unwrap();

    let edge = LineageEdgeV1 {
        input_fingerprint_v0: "a".repeat(64),
        output_fingerprint_v0: "b".repeat(64),
        node_id: node_b_id,
        op_kind: OpKind::Data,
    };
    bundle
        .write_dataset_lineage(&DatasetLineageV1 {
            schema_version: 1,
            edges: vec![edge.clone()],
        })
        .unwrap();
    bundle.append_lineage_edge_update(&edge).unwrap();

    // Test lossless u64::MAX nanosecond precision (> 2^53 - 1).
    let mut attrs = AttrMap::new();
    attrs.insert("round".to_string(), AttrValue::U64(u64::MAX));
    let span = SpanRecord {
        schema_version: 1,
        trace_id: TraceId::from_bytes([0x11; 16]),
        span_id: SpanId::from_bytes([0x22; 8]),
        parent_span_id: None,
        name: "round_step".to_string(),
        start_unix_nanos: u64::MAX - 1,
        end_unix_nanos: Some(u64::MAX),
        attrs: attrs.clone(),
    };
    bundle.append_span(&span).unwrap();

    let event = EventRecord {
        schema_version: 1,
        ts_unix_nanos: u64::MAX,
        trace_id: TraceId::from_bytes([0x11; 16]),
        span_id: Some(SpanId::from_bytes([0x22; 8])),
        name: "quorum_committed".to_string(),
        attrs: attrs.clone(),
    };
    bundle.append_event(&event).unwrap();

    let metric = MetricRecord {
        schema_version: 1,
        ts_unix_nanos: u64::MAX,
        trace_id: TraceId::from_bytes([0x11; 16]),
        span_id: None,
        name: "loss".to_string(),
        value: 0.125,
        unit: Some("nats".to_string()),
        attrs,
    };
    bundle.append_metric(&metric).unwrap();

    let mat_v1 = MaterializationRecordV1 {
        schema_version: 1,
        ts_unix_nanos: u64::MAX - 10,
        asset_key: "dataset://ns/raw".to_string(),
        fingerprint_v0: "a".repeat(64),
        node_id: node_a_id,
        node_def_hash: "e".repeat(64),
        rows: Some(100),
        bytes: Some(4096),
        cache_hit: Some(false),
        duration_ms: Some(12),
        quality_flags: None,
        unsafe_surface: false,
    };
    bundle.append_materialization(&mat_v1).unwrap();

    let mat_v2 = MaterializationRecordV2 {
        schema_version: MATERIALIZATION_SCHEMA_V2,
        record_seq: u64::MAX,
        ts_unix_nanos: u64::MAX,
        asset_key: "dataset://ns/clean".to_string(),
        fingerprint_v0: "b".repeat(64),
        node_id: node_b_id,
        node_def_hash: "f".repeat(64),
        op_type: "transform".to_string(),
        input_asset_keys: vec!["dataset://ns/raw".to_string()],
        input_fingerprints_v0: vec!["a".repeat(64)],
        rows: Some(100),
        bytes: Some(4096),
        duration_ms: Some(7),
        cache_decision: CacheDecisionV0::Hit,
        cache_reason: Some("cas_verified".to_string()),
        cache_key_v0: Some("9".repeat(64)),
        cache_hit: Some(true),
        unsafe_surface: false,
        unsafe_reasons: vec![UnsafeReasonV0::UntrustedInput],
        applied_transforms: vec![TransformAuditV0 {
            transform_name: "clip_l2".to_string(),
            core_trusted: true,
            round_id: u64::MAX,
        }],
        status: MaterializationStatusV0::Ok,
        error_code: None,
        quality: None,
    };
    bundle.append_materialization_v2(&mat_v2).unwrap();

    let signer = KeyPair::from_seed([0x55; 32]).unwrap();
    bundle.finalize_signed_manifest(&signer).unwrap();

    let policy = TrustPolicy::Strict {
        trusted_public_keys: vec![*signer.public_key()],
        expected_run_id: Some(run_id),
        min_signed_unix_nanos: Some(1),
    };
    let verified = bundle.verify_and_load(&policy).unwrap();
    validate_verified_bundle_schemas(&verified)
        .expect("all verified bundle files must conform to Draft 2020-12 schemas");

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn schema_validator_rejects_negative_fixtures_across_contracts() {
    // 1. manifest.schema.json: invalid sha256 hex length
    let bad_manifest = br#"{
        "schema_version": 1,
        "run_id": "01010101010101010101010101010101",
        "hash_algo": "sha256",
        "entries": [
            { "path": "run.json", "sha256": "deadbeef", "bytes": 10, "required": true }
        ]
    }"#;
    assert!(validate_json_bytes_against_schema(bad_manifest, MANIFEST_SCHEMA_JSON).is_err());

    // 2. run.schema.json: unexpected additional property
    let bad_run = br#"{
        "schema_version": 1,
        "run_id": "01010101010101010101010101010101",
        "created_unix_nanos": 100,
        "swarmtorch_version": "0.1.0",
        "profile": "std",
        "injected_field": true
    }"#;
    assert!(validate_json_bytes_against_schema(bad_run, RUN_SCHEMA_JSON).is_err());

    // 3. span.schema.json: negative timestamp
    let bad_span = br#"{"schema_version":1,"trace_id":"11111111111111111111111111111111","span_id":"2222222222222222","parent_span_id":null,"name":"s","start_unix_nanos":-1,"end_unix_nanos":null,"attrs":{}}"#;
    assert!(validate_ndjson_bytes_against_schema(bad_span, SPAN_SCHEMA_JSON).is_err());

    // 4. materialization.schema.json: unknown schema_version 99
    let bad_mat = br#"{"schema_version":99,"ts_unix_nanos":1,"asset_key":"k","fingerprint_v0":"a","node_id":"11111111111111111111111111111111","node_def_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","unsafe_surface":false}"#;
    assert!(validate_ndjson_bytes_against_schema(bad_mat, MATERIALIZATION_SCHEMA_JSON).is_err());
}

#[test]
fn packaged_schemas_match_workspace_root_schemas() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_schemas = manifest_dir.join("..").join("schemas").join("v1");
    if !workspace_schemas.is_dir() {
        // Running inside an isolated `cargo package` verification directory where `../schemas/v1` is absent.
        return;
    }

    let schemas = [
        ("manifest.schema.json", MANIFEST_SCHEMA_JSON),
        ("run.schema.json", RUN_SCHEMA_JSON),
        ("graph.schema.json", GRAPH_SCHEMA_JSON),
        ("registry.schema.json", REGISTRY_SCHEMA_JSON),
        ("lineage.schema.json", LINEAGE_SCHEMA_JSON),
        ("span.schema.json", SPAN_SCHEMA_JSON),
        ("event.schema.json", EVENT_SCHEMA_JSON),
        ("metric.schema.json", METRIC_SCHEMA_JSON),
        ("materialization.schema.json", MATERIALIZATION_SCHEMA_JSON),
    ];

    for (name, embedded) in schemas {
        let root_path = workspace_schemas.join(name);
        let root_raw = fs::read_to_string(&root_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", root_path.display()));
        assert_eq!(
            embedded.replace("\r\n", "\n"),
            root_raw.replace("\r\n", "\n"),
            "packaged schema {name} drifted from workspace root schemas/v1/{name}"
        );
    }
}
