//! End-to-end artifact pipeline demo (Phase 2, alpha.6)
//!
//! Demonstrates:
//! 1. Source registration → transforms → finalize → report generation
//! 2. CoreOnlyPolicy denial of UnsafeExtension + PermissivePolicy allowance
//! 3. Trust propagation: untrusted input or non-Core trust → unsafe_surface
//! 4. Multi-output fingerprint uniqueness
//! 5. Manifest validation round-trip
//!
//! Run: `cargo run -p swarm-torch --example artifact_pipeline`

use std::sync::Arc;

use swarm_torch::artifacts::{DataOpsSession, OutputSpec, RunArtifactBundle, RunArtifactSink};
use swarm_torch::report;

use swarm_torch_core::dataops::{
    DatasetRegistryV1, SchemaDescriptorV0, SourceDescriptorV0, TrustClass,
};
use swarm_torch_core::execution::{
    CoreOnlyPolicy, ExecutionPolicy, PermissivePolicy, PolicyDecision,
};
use swarm_torch_core::observe::RunId;
use swarm_torch_core::run_graph::{
    AssetRefV1, CanonParams, ExecutionTrust, GraphV1, NodeV1, OpKind,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn make_node(
    key: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    trust: ExecutionTrust,
) -> NodeV1 {
    NodeV1 {
        node_key: key.to_string(),
        node_id: None,
        op_kind: OpKind::Data,
        op_type: op_type.to_string(),
        inputs: inputs
            .iter()
            .map(|k| AssetRefV1 {
                asset_key: k.to_string(),
                fingerprint: None,
            })
            .collect(),
        outputs: outputs
            .iter()
            .map(|k| AssetRefV1 {
                asset_key: k.to_string(),
                fingerprint: None,
            })
            .collect(),
        params: CanonParams::new(),
        code_ref: None,
        unsafe_surface: false,
        execution_trust: trust,
        node_def_hash: None,
    }
}

fn main() {
    println!("SwarmTorch Artifact Pipeline Demo");
    println!("=================================\n");

    // --- Set up temp dir and bundle ---
    let base = std::env::temp_dir().join("swarmtorch_pipeline_demo");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0xAB; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
    println!("✓ Bundle created: {}", bundle.run_dir().display());

    // --- Build graph nodes ---
    let ingest_node = make_node(
        "ingest/raw_data",
        "ingest",
        &[],
        &["dataset://demo/raw"],
        ExecutionTrust::Core,
    );

    let core_transform = make_node(
        "prep/clean",
        "transform",
        &["dataset://demo/raw"],
        &["dataset://demo/clean"],
        ExecutionTrust::Core,
    );

    // This node has UnsafeExtension trust — CoreOnlyPolicy will DENY it
    let unsafe_transform = make_node(
        "plugin/enrich",
        "transform",
        &["dataset://demo/clean"],
        &[
            "dataset://demo/enriched_left",
            "dataset://demo/enriched_right",
        ],
        ExecutionTrust::UnsafeExtension,
    );

    // --- Write graph.json ---
    let graph = GraphV1 {
        nodes: vec![
            ingest_node.clone(),
            core_transform.clone(),
            unsafe_transform.clone(),
        ],
        ..Default::default()
    };

    let sink = Arc::new(RunArtifactSink::new(bundle));
    sink.write_graph(&graph).unwrap();
    println!("✓ graph.json written ({} nodes)", graph.nodes.len());

    // --- DataOps session ---
    let mut session = DataOpsSession::new(Arc::clone(&sink));

    // 1. Register source dataset as Trusted
    let source = SourceDescriptorV0 {
        uri: "s3://demo-bucket/raw.parquet".to_string(),
        content_type: "application/parquet".to_string(),
        auth_mode: swarm_torch_core::dataops::AuthModeMarker::BearerToken,
        etag_or_version: Some("v1".to_string()),
    };
    let schema = SchemaDescriptorV0 {
        format: "arrow-json".to_string(),
        canonical: r#"{"fields":[{"name":"id","type":"i64"},{"name":"value","type":"f64"}]}"#
            .to_string(),
    };

    session
        .register_source(
            "dataset://demo/raw",
            TrustClass::Trusted,
            source,
            Some(schema),
            &ingest_node,
        )
        .unwrap();
    println!("✓ Source registered: dataset://demo/raw (Trusted)");

    // --- Policy checks ---
    let empty_registry = DatasetRegistryV1::default();

    // 2. CoreOnlyPolicy allows Core transform
    let core_policy = CoreOnlyPolicy;
    let decision = core_policy.allow(&core_transform, &empty_registry);
    assert_eq!(decision, PolicyDecision::Allowed);
    println!("✓ CoreOnlyPolicy: ALLOWED prep/clean (Core trust)");

    // 3. CoreOnlyPolicy DENIES UnsafeExtension
    let decision = core_policy.allow(&unsafe_transform, &empty_registry);
    match &decision {
        PolicyDecision::Denied { reason } => {
            println!("✓ CoreOnlyPolicy: DENIED plugin/enrich — {reason}");
        }
        PolicyDecision::Allowed => panic!("CoreOnlyPolicy should deny UnsafeExtension"),
    }

    // 4. PermissivePolicy allows it (for dev/testing)
    let permissive = PermissivePolicy;
    let decision = permissive.allow(&unsafe_transform, &empty_registry);
    assert_eq!(decision, PolicyDecision::Allowed);
    println!("✓ PermissivePolicy: ALLOWED plugin/enrich (UnsafeExtension)");

    // --- Execute transforms (under PermissivePolicy) ---

    // 5. Core transform: output should be Trusted
    session
        .materialize_node_outputs(
            &core_transform,
            &[OutputSpec {
                asset_key: "dataset://demo/clean".to_string(),
                schema: None,
                rows: Some(10_000),
                bytes: Some(500_000),
            }],
            1_000_000_000,
            false,
            150,
        )
        .unwrap();
    println!("✓ Materialized: dataset://demo/clean (Core → Trusted)");

    // 6. Unsafe transform: produces 2 outputs, both should be Untrusted
    session
        .materialize_node_outputs(
            &unsafe_transform,
            &[
                OutputSpec {
                    asset_key: "dataset://demo/enriched_left".to_string(),
                    schema: None,
                    rows: Some(5_000),
                    bytes: Some(250_000),
                },
                OutputSpec {
                    asset_key: "dataset://demo/enriched_right".to_string(),
                    schema: None,
                    rows: Some(5_000),
                    bytes: Some(250_000),
                },
            ],
            2_000_000_000,
            false,
            300,
        )
        .unwrap();
    println!("✓ Materialized: enriched_left + enriched_right (UnsafeExtension → Untrusted)");

    // --- Verify fingerprint uniqueness ---
    let fp_left = session
        .fingerprint("dataset://demo/enriched_left")
        .unwrap()
        .to_string();
    let fp_right = session
        .fingerprint("dataset://demo/enriched_right")
        .unwrap()
        .to_string();
    assert_ne!(
        fp_left, fp_right,
        "multi-output fingerprints must be unique"
    );
    assert_eq!(fp_left.len(), 64, "fingerprint must be 64 hex chars");
    println!("✓ Multi-output fingerprints unique:");
    println!("    left:  {fp_left}");
    println!("    right: {fp_right}");

    // --- Finalize (writes manifest) ---
    session.finalize().unwrap();
    println!("✓ Session finalized (manifest.json written)");

    // --- Generate report ---
    let html_out = base.join("report.html");
    let json_out = base.join("report.json");
    report::generate_report(sink.bundle().run_dir(), &html_out, Some(&json_out)).unwrap();
    println!("✓ Report generated:");
    println!("    HTML: {}", html_out.display());
    println!("    JSON: {}", json_out.display());

    // --- Validate report loads (proves manifest round-trip) ---
    let loaded = report::load_report(sink.bundle().run_dir()).unwrap();
    assert_eq!(loaded.graph.nodes.len(), 3, "graph should have 3 nodes");
    assert_eq!(
        loaded.registry.datasets.len(),
        4,
        "registry should have 4 datasets"
    );
    assert_eq!(
        loaded.materializations.len(),
        3,
        "should have 3 materializations"
    );
    println!("✓ Report loads + manifest valid:");
    println!("    Nodes: {}", loaded.graph.nodes.len());
    println!("    Datasets: {}", loaded.registry.datasets.len());
    println!("    Materializations: {}", loaded.materializations.len());

    // --- Verify unsafe surfaces ---
    let clean_mat = loaded
        .materializations
        .iter()
        .find(|m| m.asset_key == "dataset://demo/clean")
        .unwrap();
    assert!(
        !clean_mat.unsafe_surface,
        "Core → Trusted should not be unsafe"
    );

    let left_mat = loaded
        .materializations
        .iter()
        .find(|m| m.asset_key == "dataset://demo/enriched_left")
        .unwrap();
    assert!(
        left_mat.unsafe_surface,
        "UnsafeExtension should mark unsafe_surface"
    );

    let right_mat = loaded
        .materializations
        .iter()
        .find(|m| m.asset_key == "dataset://demo/enriched_right")
        .unwrap();
    assert!(
        right_mat.unsafe_surface,
        "UnsafeExtension should mark unsafe_surface"
    );
    println!("✓ Unsafe surface flags correct:");
    println!("    clean: unsafe={}", clean_mat.unsafe_surface);
    println!("    enriched_left: unsafe={}", left_mat.unsafe_surface);
    println!("    enriched_right: unsafe={}", right_mat.unsafe_surface);

    // --- Verify lineage ---
    assert!(
        loaded.lineage.edges.len() >= 2,
        "should have at least 2 lineage edges"
    );
    println!("✓ Lineage edges: {}", loaded.lineage.edges.len());

    // --- Cleanup ---
    let _ = std::fs::remove_dir_all(&base);

    println!("\n================================");
    println!("All pipeline assertions passed! ✓");
}
