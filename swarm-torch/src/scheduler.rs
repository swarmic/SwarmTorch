//! Sequential graph scheduler (Wave 7 / Wave 10).
//!
//! This module provides a deterministic, single-process executor for `GraphV1`:
//! - topological scheduling with cycle detection
//! - policy checks before runner invocation
//! - fail-closed handling for missing inputs
//! - verified cache-hit short-circuiting (`ADR-0023` §2.3, `W10-01`)
//! - preservation of runner `Vec<AssetInstanceV1>` outputs and physical CAS URIs
//! - status materialization records for failed/skipped nodes

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;

use swarm_torch_core::dataops::{CacheDecisionV0, OutputSpecCore};
use swarm_torch_core::execution::{AssetInstanceV1, ExecutionPolicy, PolicyDecision};
use swarm_torch_core::observe::{AttrMap, AttrValue, EventRecord, RunEventEmitter, RunId, TraceId};
use swarm_torch_core::run_graph::{
    node_id_from_key, validate_graph_v1, GraphV1, GraphValidationError, NodeId, NodeV1,
};

use crate::artifacts::{parse_cas_uri_digest, DataOpsSession, OutputSpec};
use crate::native_runner::{ExecutionContext, NativeOpRunner};

#[derive(Debug)]
pub enum SchedulerError {
    GraphNormalization(String),
    GraphValidation(GraphValidationError),
    InvalidEdgeReference {
        from_node_id: NodeId,
        to_node_id: NodeId,
    },
    CycleDetected {
        node_keys: Vec<String>,
    },
    InvariantViolation(&'static str),
    Io(io::Error),
}

impl core::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::GraphNormalization(message) => write!(f, "graph normalization failed: {message}"),
            Self::GraphValidation(error) => write!(f, "graph validation failed: {error}"),
            Self::InvalidEdgeReference {
                from_node_id,
                to_node_id,
            } => write!(
                f,
                "graph edge references unknown node_id: {} -> {}",
                from_node_id, to_node_id
            ),
            Self::CycleDetected { node_keys } => {
                write!(f, "graph contains cycle(s): {}", node_keys.join(","))
            }
            Self::InvariantViolation(message) => {
                write!(f, "scheduler invariant violation: {message}")
            }
            Self::Io(error) => write!(f, "scheduler I/O error: {error}"),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl From<io::Error> for SchedulerError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<GraphValidationError> for SchedulerError {
    fn from(value: GraphValidationError) -> Self {
        Self::GraphValidation(value)
    }
}

/// Execution isolation profile for graph node runners (`ADR-0023` §3, Wave 12 `W12-04` stub).
///
/// In Wave 11 (`v0.1.0`), `execute_graph_sequential` executes `ExecutionTrust::Core` nodes via
/// [`RunnerIsolationProfile::InProcessTrusted`]. Wave 12 (`W12-04`) will wire
/// [`RunnerIsolationProfile::ExternalSandboxedStub`] to OS-level subprocess/Wasm isolation for
/// `ExecutionTrust::SandboxedExtension` and `ExecutionTrust::UntrustedExternal` nodes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum RunnerIsolationProfile {
    /// Execute trusted core operators in-process (`ExecutionTrust::Core`).
    #[default]
    InProcessTrusted,
    /// Wave 12 (`W12-04`) real deterministic WebAssembly sandboxed executor (`ExecutionTrust::SandboxedExtension`).
    WasmSandboxedV1(crate::wasm_sandbox::SandboxedExtensionPolicyV1),
    /// Legacy placeholder stub for external process isolation.
    ExternalSandboxedStub {
        /// Maximum wall-clock execution time in milliseconds.
        timeout_ms: u64,
        /// Maximum resident memory in bytes.
        max_memory_bytes: u64,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SchedulerReport {
    pub executed_nodes: Vec<String>,
    pub cached_nodes: Vec<String>,
    pub failed_nodes: Vec<String>,
    pub skipped_nodes: Vec<String>,
    pub runner_outputs: BTreeMap<String, Vec<AssetInstanceV1>>,
}

/// Return nodes in deterministic topological order (`node_key` tie-break).
pub fn topological_sort_nodes(graph: &GraphV1) -> Result<Vec<NodeV1>, SchedulerError> {
    validate_graph_v1(graph)?;

    let normalized = graph
        .clone()
        .normalize()
        .map_err(|e| SchedulerError::GraphNormalization(e.to_string()))?;
    validate_graph_v1(&normalized)?;

    let mut node_by_id: HashMap<String, NodeV1> = HashMap::new();
    for mut node in normalized.nodes {
        let node_id = node
            .node_id
            .unwrap_or_else(|| node_id_from_key(&node.node_key));
        node.node_id = Some(node_id);
        let node_id_hex = node_id.to_string();
        if let Some(existing) = node_by_id.get(&node_id_hex) {
            return Err(SchedulerError::GraphValidation(
                GraphValidationError::DuplicateNodeId {
                    node_id: node_id_hex,
                    first_node_key: existing.node_key.clone(),
                    duplicate_node_key: node.node_key.clone(),
                },
            ));
        }
        node_by_id.insert(node_id_hex, node);
    }

    let mut indegree: HashMap<String, usize> = HashMap::new();
    let mut adjacency: HashMap<String, BTreeSet<String>> = HashMap::new();
    for node_id in node_by_id.keys() {
        indegree.insert(node_id.clone(), 0);
        adjacency.insert(node_id.clone(), BTreeSet::new());
    }

    for edge in normalized.edges {
        let from_id = edge.from_node_id.to_string();
        let to_id = edge.to_node_id.to_string();
        if !node_by_id.contains_key(&from_id) || !node_by_id.contains_key(&to_id) {
            return Err(SchedulerError::InvalidEdgeReference {
                from_node_id: edge.from_node_id,
                to_node_id: edge.to_node_id,
            });
        }
        let from_neighbors =
            adjacency
                .get_mut(&from_id)
                .ok_or(SchedulerError::InvariantViolation(
                    "missing adjacency set for from node_id",
                ))?;
        let edge_inserted = from_neighbors.insert(to_id.clone());
        if edge_inserted {
            let to_degree = indegree
                .get_mut(&to_id)
                .ok_or(SchedulerError::InvariantViolation(
                    "missing indegree entry for to node_id",
                ))?;
            *to_degree += 1;
        }
    }

    let mut ready: BTreeSet<(String, String)> = BTreeSet::new();
    for (node_id, degree) in &indegree {
        if *degree == 0 {
            let node = node_by_id
                .get(node_id)
                .ok_or(SchedulerError::InvariantViolation(
                    "missing node for indegree entry",
                ))?;
            ready.insert((node.node_key.clone(), node_id.clone()));
        }
    }

    let mut ordered = Vec::with_capacity(node_by_id.len());
    while let Some((_, node_id)) = ready.pop_first() {
        let node = node_by_id
            .get(&node_id)
            .ok_or(SchedulerError::InvariantViolation(
                "missing node for ready queue entry",
            ))?
            .clone();
        ordered.push(node);

        let neighbors: Vec<String> = adjacency
            .get(&node_id)
            .ok_or(SchedulerError::InvariantViolation(
                "missing adjacency for ready queue entry",
            ))?
            .iter()
            .cloned()
            .collect();
        for neighbor in neighbors {
            let degree = indegree
                .get_mut(&neighbor)
                .ok_or(SchedulerError::InvariantViolation(
                    "missing indegree for adjacency neighbor",
                ))?;
            *degree = degree.saturating_sub(1);
            if *degree == 0 {
                let neighbor_key = node_by_id
                    .get(&neighbor)
                    .ok_or(SchedulerError::InvariantViolation(
                        "missing node for adjacency neighbor",
                    ))?
                    .node_key
                    .clone();
                ready.insert((neighbor_key, neighbor.clone()));
            }
        }
    }

    if ordered.len() != node_by_id.len() {
        let mut remaining: Vec<String> = Vec::new();
        for (node_id, degree) in &indegree {
            if *degree > 0 {
                let node = node_by_id
                    .get(node_id)
                    .ok_or(SchedulerError::InvariantViolation(
                        "missing node while collecting cycle members",
                    ))?;
                remaining.push(node.node_key.clone());
            }
        }
        remaining.sort();
        return Err(SchedulerError::CycleDetected {
            node_keys: remaining,
        });
    }

    Ok(ordered)
}

/// Execute a graph sequentially with fail-closed semantics and verified cache-hit reuse.
pub fn execute_graph_sequential(
    graph: &GraphV1,
    session: &mut DataOpsSession,
    runner: &NativeOpRunner,
    policy: &dyn ExecutionPolicy,
    run_id: RunId,
    clock_nanos: fn() -> u64,
) -> Result<SchedulerReport, SchedulerError> {
    execute_graph_with_isolation(
        graph,
        session,
        runner,
        policy,
        &RunnerIsolationProfile::InProcessTrusted,
        None,
        run_id,
        clock_nanos,
    )
}

/// Execute a graph sequentially under an explicit [`RunnerIsolationProfile`] and optional pinned
/// WebAssembly extension registry (`ADR-0018`, `W12-04`).
#[allow(clippy::too_many_arguments)]
pub fn execute_graph_with_isolation(
    graph: &GraphV1,
    session: &mut DataOpsSession,
    runner: &NativeOpRunner,
    policy: &dyn ExecutionPolicy,
    isolation: &RunnerIsolationProfile,
    wasm_registry: Option<&std::collections::BTreeMap<[u8; 32], Vec<u8>>>,
    run_id: RunId,
    clock_nanos: fn() -> u64,
) -> Result<SchedulerReport, SchedulerError> {
    let empty_registry = std::collections::BTreeMap::new();
    let ordered = topological_sort_nodes(graph)?;
    let mut report = SchedulerReport::default();
    let trace_id = TraceId::from_bytes(*run_id.as_bytes());
    let context = ExecutionContext {
        run_id,
        clock_nanos,
    };

    for node in ordered {
        let started = (clock_nanos)();
        let registry = session.registry_snapshot();
        match policy.allow(&node, &registry) {
            PolicyDecision::Allowed => {}
            PolicyDecision::Denied { reason } => {
                let error_code = format!("policy_denied:{reason}");
                session.record_node_error(&node, started, 0, error_code.clone())?;
                emit_scheduler_event(
                    session.sink().as_ref(),
                    trace_id,
                    started,
                    "scheduler/node_denied",
                    &node.node_key,
                    Some(&error_code),
                )?;
                report.failed_nodes.push(node.node_key.clone());
                continue;
            }
        }

        let mut inputs: Vec<AssetInstanceV1> = Vec::with_capacity(node.inputs.len());
        let mut missing_input: Option<String> = None;
        for input in &node.inputs {
            if let Some(instance) = session.resolve_asset_instance(&input.asset_key) {
                inputs.push(instance);
            } else {
                missing_input = Some(input.asset_key.clone());
                break;
            }
        }
        if let Some(asset_key) = missing_input {
            let error_code = format!("missing_input:{asset_key}");
            session.record_node_skipped(&node, started, error_code.clone())?;
            emit_scheduler_event(
                session.sink().as_ref(),
                trace_id,
                started,
                "scheduler/node_skipped",
                &node.node_key,
                Some(&error_code),
            )?;
            report.skipped_nodes.push(node.node_key.clone());
            continue;
        }

        let core_specs: Vec<OutputSpecCore> = node
            .outputs
            .iter()
            .map(|output| OutputSpecCore {
                asset_key: output.asset_key.clone(),
                schema: None,
            })
            .collect();
        let mut output_specs: Vec<OutputSpec> = node
            .outputs
            .iter()
            .map(|output| OutputSpec {
                asset_key: output.asset_key.clone(),
                schema: None,
                rows: None,
                bytes: None,
            })
            .collect();

        let cache_decision = session.verify_cache_hit(&node, &core_specs);
        if cache_decision == CacheDecisionV0::Hit {
            let finished = (clock_nanos)();
            if let Err(error) = session.materialize_node_outputs(
                &node,
                &output_specs,
                finished,
                CacheDecisionV0::Hit,
                0,
            ) {
                let error_code = format!("materialization_error:{error}");
                session.record_node_error(&node, finished, 0, error_code.clone())?;
                emit_scheduler_event(
                    session.sink().as_ref(),
                    trace_id,
                    finished,
                    "scheduler/node_failed",
                    &node.node_key,
                    Some(&error_code),
                )?;
                report.failed_nodes.push(node.node_key.clone());
                continue;
            }

            let resolved_outputs: Vec<AssetInstanceV1> = node
                .outputs
                .iter()
                .filter_map(|o| session.resolve_asset_instance(&o.asset_key))
                .collect();
            report
                .runner_outputs
                .insert(node.node_key.clone(), resolved_outputs);
            emit_scheduler_event(
                session.sink().as_ref(),
                trace_id,
                finished,
                "scheduler/node_cached",
                &node.node_key,
                Some("cache_hit"),
            )?;
            emit_scheduler_event(
                session.sink().as_ref(),
                trace_id,
                finished,
                "scheduler/node_succeeded",
                &node.node_key,
                None,
            )?;
            report.cached_nodes.push(node.node_key.clone());
            report.executed_nodes.push(node.node_key.clone());
            continue;
        }

        // Fail-closed isolation enforcement before invoking runner
        let wasm_arg = match (&node.execution_trust, isolation) {
            (swarm_torch_core::run_graph::ExecutionTrust::Core, _) => {
                if node.op_type == "wasm_transform" {
                    let error_code =
                        "isolation_denied:wasm_transform requires SandboxedExtension trust"
                            .to_string();
                    session.record_node_error(&node, started, 0, error_code.clone())?;
                    emit_scheduler_event(
                        session.sink().as_ref(),
                        trace_id,
                        started,
                        "scheduler/node_denied",
                        &node.node_key,
                        Some(&error_code),
                    )?;
                    report.failed_nodes.push(node.node_key.clone());
                    continue;
                }
                None
            }
            (
                swarm_torch_core::run_graph::ExecutionTrust::SandboxedExtension,
                RunnerIsolationProfile::WasmSandboxedV1(wasm_policy),
            ) => Some((wasm_policy, wasm_registry.unwrap_or(&empty_registry))),
            (swarm_torch_core::run_graph::ExecutionTrust::SandboxedExtension, _) => {
                let error_code =
                    "isolation_denied:SandboxedExtension requires RunnerIsolationProfile::WasmSandboxedV1"
                        .to_string();
                session.record_node_error(&node, started, 0, error_code.clone())?;
                emit_scheduler_event(
                    session.sink().as_ref(),
                    trace_id,
                    started,
                    "scheduler/node_denied",
                    &node.node_key,
                    Some(&error_code),
                )?;
                report.failed_nodes.push(node.node_key.clone());
                continue;
            }
            (swarm_torch_core::run_graph::ExecutionTrust::UnsafeExtension, _) => {
                let error_code =
                    "isolation_denied:UnsafeExtension has no permitted host executor".to_string();
                session.record_node_error(&node, started, 0, error_code.clone())?;
                emit_scheduler_event(
                    session.sink().as_ref(),
                    trace_id,
                    started,
                    "scheduler/node_denied",
                    &node.node_key,
                    Some(&error_code),
                )?;
                report.failed_nodes.push(node.node_key.clone());
                continue;
            }
        };

        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let runner_result = runner.run_in_session(
            &context,
            &run_dir,
            &node,
            &inputs,
            session.sink().as_ref(),
            wasm_arg,
        );
        let finished = (clock_nanos)();
        let duration_ms = finished.saturating_sub(started) / 1_000_000;
        let exec_outcome = match runner_result {
            Ok(outcome) => outcome,
            Err(error) => {
                let error_code = format!("runner_error:{error}");
                session.record_node_error(&node, finished, duration_ms, error_code.clone())?;
                emit_scheduler_event(
                    session.sink().as_ref(),
                    trace_id,
                    finished,
                    "scheduler/node_failed",
                    &node.node_key,
                    Some(&error_code),
                )?;
                report.failed_nodes.push(node.node_key.clone());
                continue;
            }
        };
        for phys_spec in &exec_outcome.output_specs {
            if let Some(target_spec) = output_specs
                .iter_mut()
                .find(|s| s.asset_key == phys_spec.asset_key)
            {
                if phys_spec.rows.is_some() {
                    target_spec.rows = phys_spec.rows;
                }
                if phys_spec.bytes.is_some() {
                    target_spec.bytes = phys_spec.bytes;
                }
            }
        }
        let runner_instances = exec_outcome.instances;

        let materialize = session.materialize_node_outputs(
            &node,
            &output_specs,
            finished,
            cache_decision,
            duration_ms,
        );
        if let Err(error) = materialize {
            let error_code = format!("materialization_error:{error}");
            session.record_node_error(&node, finished, duration_ms, error_code.clone())?;
            emit_scheduler_event(
                session.sink().as_ref(),
                trace_id,
                finished,
                "scheduler/node_failed",
                &node.node_key,
                Some(&error_code),
            )?;
            report.failed_nodes.push(node.node_key.clone());
            continue;
        }

        // Preserve any verified CAS URI returned by the runner on declared outputs.
        for instance in &runner_instances {
            if let Some(ref uri) = instance.uri {
                if parse_cas_uri_digest(uri).is_some()
                    && node
                        .outputs
                        .iter()
                        .any(|declared| declared.asset_key == instance.asset_key)
                {
                    session.bind_output_cas_uri(
                        &instance.asset_key,
                        uri,
                        "application/octet-stream",
                    )?;
                }
            }
        }
        report
            .runner_outputs
            .insert(node.node_key.clone(), runner_instances);

        emit_scheduler_event(
            session.sink().as_ref(),
            trace_id,
            finished,
            "scheduler/node_succeeded",
            &node.node_key,
            None,
        )?;
        report.executed_nodes.push(node.node_key.clone());
    }

    Ok(report)
}

fn emit_scheduler_event(
    sink: &crate::artifacts::RunArtifactSink,
    trace_id: TraceId,
    ts_unix_nanos: u64,
    name: &str,
    node_key: &str,
    detail: Option<&str>,
) -> io::Result<()> {
    let mut attrs = AttrMap::new();
    attrs.insert(
        "swarmtorch.node_key".to_string(),
        AttrValue::Str(node_key.to_string()),
    );
    if let Some(detail) = detail {
        attrs.insert(
            "swarmtorch.detail".to_string(),
            AttrValue::Str(detail.to_string()),
        );
    }
    let event = EventRecord {
        schema_version: 1,
        ts_unix_nanos,
        trace_id,
        span_id: None,
        name: name.to_string(),
        attrs,
    };
    sink.emit_event(&event)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use swarm_torch_core::dataops::{MaterializationStatusV0, SourceDescriptorV0, TrustClass};
    use swarm_torch_core::execution::{CoreOnlyPolicy, PermissivePolicy};
    use swarm_torch_core::run_graph::{
        AssetRefV1, CachePolicyV0, CanonParams, EdgeV1, ExecutionTrust, OpKind,
    };

    use crate::artifacts::{RunArtifactBundle, RunArtifactSink};
    use crate::report::load_report;

    fn temp_dir(prefix: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("swarmtorch_scheduler_{prefix}_{pid}_{nanos}"));
        dir
    }

    fn test_clock() -> u64 {
        static CLOCK: AtomicU64 = AtomicU64::new(1_000_000_000);
        CLOCK.fetch_add(1_000_000, Ordering::SeqCst)
    }

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
                .map(|asset_key| AssetRefV1 {
                    asset_key: asset_key.to_string(),
                    fingerprint: None,
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|asset_key| AssetRefV1 {
                    asset_key: asset_key.to_string(),
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

    fn create_session(prefix: &str) -> (PathBuf, DataOpsSession) {
        let base = temp_dir(prefix);
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([0x42; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        let sink = Arc::new(RunArtifactSink::new(bundle));
        (base, DataOpsSession::new(sink))
    }

    fn register_source(session: &mut DataOpsSession, asset_key: &str) {
        let ingest = make_node(
            "ingest/raw",
            "ingest",
            &[],
            &[asset_key],
            ExecutionTrust::Core,
        );
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/raw.parquet".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
            etag_or_version: Some("v1".to_string()),
        };
        session
            .register_source(asset_key, TrustClass::Trusted, source, None, &ingest)
            .unwrap();
    }

    #[test]
    fn scheduler_rejects_graph_cycles() {
        let mut graph = GraphV1::default();
        let a = make_node(
            "node/a",
            "passthrough",
            &[],
            &["dataset://ns/a"],
            ExecutionTrust::Core,
        );
        let b = make_node(
            "node/b",
            "passthrough",
            &["dataset://ns/a"],
            &["dataset://ns/b"],
            ExecutionTrust::Core,
        );
        let na = node_id_from_key(&a.node_key);
        let nb = node_id_from_key(&b.node_key);
        graph.nodes = vec![a, b];
        graph.edges = vec![
            EdgeV1 {
                from_node_id: na,
                to_node_id: nb,
                asset_key: None,
            },
            EdgeV1 {
                from_node_id: nb,
                to_node_id: na,
                asset_key: None,
            },
        ];

        let err = topological_sort_nodes(&graph).expect_err("cycle should be rejected");
        assert!(matches!(err, SchedulerError::CycleDetected { .. }));
    }

    #[test]
    fn scheduler_rejects_duplicate_node_ids() {
        let mut a = make_node(
            "node/a",
            "passthrough",
            &[],
            &["dataset://ns/a"],
            ExecutionTrust::Core,
        );
        let mut b = make_node(
            "node/b",
            "passthrough",
            &[],
            &["dataset://ns/b"],
            ExecutionTrust::Core,
        );
        let duplicate = node_id_from_key("node/shared");
        a.node_id = Some(duplicate);
        b.node_id = Some(duplicate);

        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("dup-ids".to_string()),
            nodes: vec![a, b],
            edges: vec![],
        };

        let err = topological_sort_nodes(&graph).expect_err("duplicate node ids should fail");
        assert!(matches!(
            err,
            SchedulerError::GraphValidation(GraphValidationError::DuplicateNodeId { .. })
        ));
    }

    #[test]
    fn scheduler_rejects_duplicate_node_keys() {
        let a = make_node(
            "node/dup",
            "passthrough",
            &[],
            &["dataset://ns/a"],
            ExecutionTrust::Core,
        );
        let b = make_node(
            "node/dup",
            "passthrough",
            &[],
            &["dataset://ns/b"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("dup-keys".to_string()),
            nodes: vec![a, b],
            edges: vec![],
        };

        let err = topological_sort_nodes(&graph).expect_err("duplicate node keys should fail");
        assert!(matches!(
            err,
            SchedulerError::GraphValidation(GraphValidationError::DuplicateNodeKey { .. })
        ));
    }

    #[test]
    fn scheduler_invalid_graph_returns_error_not_panic() {
        let node = make_node(
            "node/only",
            "passthrough",
            &[],
            &["dataset://ns/out"],
            ExecutionTrust::Core,
        );
        let missing = node_id_from_key("node/missing");
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("invalid-edge".to_string()),
            nodes: vec![node.clone()],
            edges: vec![EdgeV1 {
                from_node_id: node_id_from_key(&node.node_key),
                to_node_id: missing,
                asset_key: None,
            }],
        };

        let err = topological_sort_nodes(&graph).expect_err("invalid edge must return typed error");
        assert!(matches!(
            err,
            SchedulerError::GraphValidation(GraphValidationError::UnknownEdgeEndpoint { .. })
        ));
    }

    #[test]
    fn scheduler_executes_in_topological_order() {
        let (base, mut session) = create_session("topological_order");
        register_source(&mut session, "dataset://ns/raw");

        let n1 = make_node(
            "node/one",
            "passthrough",
            &["dataset://ns/raw"],
            &["dataset://ns/a"],
            ExecutionTrust::Core,
        );
        let n2 = make_node(
            "node/two",
            "union",
            &["dataset://ns/a"],
            &["dataset://ns/b"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("test".to_string()),
            nodes: vec![n1.clone(), n2.clone()],
            edges: vec![EdgeV1 {
                from_node_id: node_id_from_key(&n1.node_key),
                to_node_id: node_id_from_key(&n2.node_key),
                asset_key: None,
            }],
        };

        let report = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x11; 16]),
            test_clock,
        )
        .unwrap();

        assert_eq!(report.executed_nodes, vec!["node/one", "node/two"]);
        assert!(session.fingerprint("dataset://ns/b").is_some());
        assert_eq!(report.runner_outputs.len(), 2);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scheduler_verified_cache_hit_and_cas_deletion_invalidation() {
        use crate::artifacts::put_checkpoint_model;
        use swarm_torch_models::simple::LinearModel;

        let (base, mut session) = create_session("cache_hit_and_cas_invalidation");
        register_source(&mut session, "dataset://ns/raw");

        let node = make_node(
            "node/cacheable",
            "passthrough",
            &["dataset://ns/raw"],
            &["dataset://ns/cached_out"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("cache-test".to_string()),
            nodes: vec![node.clone()],
            edges: vec![],
        };

        // First run: CacheDecisionV0::Miss
        let r1 = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x12; 16]),
            test_clock,
        )
        .unwrap();
        assert!(r1.cached_nodes.is_empty());

        // Bind a physical CAS blob to dataset://ns/cached_out
        use swarm_torch_core::traits::SwarmModel;
        let mut model = LinearModel::new(2, 1).unwrap();
        model.load_parameters(&[1.0, 2.0, 0.5]).unwrap();
        let (digest, cas_uri, _) =
            put_checkpoint_model(session.sink().bundle().run_dir(), &model).unwrap();
        session
            .bind_output_cas_uri(
                "dataset://ns/cached_out",
                &cas_uri,
                "application/octet-stream",
            )
            .unwrap();

        // Second run with CAS blob intact: CacheDecisionV0::Hit!
        let r2 = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x12; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r2.cached_nodes, vec!["node/cacheable"]);

        // Delete the physical CAS blob on disk -> must invalidate cache hit and force Miss!
        let blob_path = session
            .sink()
            .bundle()
            .run_dir()
            .join("blobs")
            .join("sha256")
            .join(digest.to_hex());
        fs::remove_file(&blob_path).unwrap();

        let r3 = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x12; 16]),
            test_clock,
        )
        .unwrap();
        assert!(
            r3.cached_nodes.is_empty(),
            "deleting physical CAS blob must force CacheDecisionV0::Miss"
        );

        // CachePolicyV0::Bypass -> must force CacheDecisionV0::Bypass even when fingerprint matches!
        let mut bypass_node = node;
        bypass_node.cache_policy = Some(CachePolicyV0::Bypass);
        let bypass_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("bypass-test".to_string()),
            nodes: vec![bypass_node],
            edges: vec![],
        };
        let r4 = execute_graph_sequential(
            &bypass_graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x12; 16]),
            test_clock,
        )
        .unwrap();
        assert!(r4.cached_nodes.is_empty());

        session.finalize().unwrap();
        let loaded = load_report(session.sink().bundle().run_dir()).unwrap();
        assert_eq!(loaded.materializations.len(), 4);
        assert_eq!(
            loaded.materializations[0].cache_decision,
            CacheDecisionV0::Miss
        );
        assert_eq!(
            loaded.materializations[1].cache_decision,
            CacheDecisionV0::Hit
        );
        assert_eq!(
            loaded.materializations[2].cache_decision,
            CacheDecisionV0::Miss
        );
        assert_eq!(
            loaded.materializations[3].cache_decision,
            CacheDecisionV0::Bypass
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scheduler_enforces_execution_policy_before_runner_call() {
        let (base, mut session) = create_session("policy_before_runner");
        register_source(&mut session, "dataset://ns/raw");

        let denied = make_node(
            "node/denied",
            "passthrough",
            &["dataset://ns/raw"],
            &["dataset://ns/out"],
            ExecutionTrust::UnsafeExtension,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("test".to_string()),
            nodes: vec![denied],
            edges: vec![],
        };

        let report = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &CoreOnlyPolicy,
            RunId::from_bytes([0x22; 16]),
            test_clock,
        )
        .unwrap();
        assert!(report.executed_nodes.is_empty());
        assert_eq!(report.failed_nodes, vec!["node/denied"]);
        assert!(session.fingerprint("dataset://ns/out").is_none());

        session.finalize().unwrap();
        let loaded = load_report(session.sink().bundle().run_dir()).unwrap();
        assert!(
            loaded
                .materializations
                .iter()
                .any(|m| m.status == MaterializationStatusV0::Error),
            "policy denial should emit error status materialization records"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scheduler_records_error_status_and_skips_dependents() {
        let (base, mut session) = create_session("error_then_skip");
        register_source(&mut session, "dataset://ns/raw");

        let failing = make_node(
            "node/failing",
            "unsupported_op",
            &["dataset://ns/raw"],
            &["dataset://ns/fail_out"],
            ExecutionTrust::Core,
        );
        let dependent = make_node(
            "node/dependent",
            "passthrough",
            &["dataset://ns/fail_out"],
            &["dataset://ns/final"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("test".to_string()),
            nodes: vec![failing.clone(), dependent.clone()],
            edges: vec![EdgeV1 {
                from_node_id: node_id_from_key(&failing.node_key),
                to_node_id: node_id_from_key(&dependent.node_key),
                asset_key: None,
            }],
        };

        let report = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x33; 16]),
            test_clock,
        )
        .unwrap();

        assert!(report.executed_nodes.is_empty());
        assert_eq!(report.failed_nodes, vec!["node/failing"]);
        assert_eq!(report.skipped_nodes, vec!["node/dependent"]);

        session.finalize().unwrap();
        let loaded = load_report(session.sink().bundle().run_dir()).unwrap();
        let statuses: Vec<MaterializationStatusV0> =
            loaded.materializations.iter().map(|m| m.status).collect();
        assert!(
            statuses.contains(&MaterializationStatusV0::Error),
            "failing node should emit error status"
        );
        assert!(
            statuses.contains(&MaterializationStatusV0::Skipped),
            "dependent node should emit skipped status"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scheduler_events_use_swarmtorch_namespace() {
        let (base, mut session) = create_session("event_namespace");
        register_source(&mut session, "dataset://ns/raw");

        let node = make_node(
            "node/a",
            "passthrough",
            &["dataset://ns/raw"],
            &["dataset://ns/a"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("event-namespace".to_string()),
            nodes: vec![node],
            edges: vec![],
        };

        let report = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x55; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(report.executed_nodes, vec!["node/a"]);

        session.finalize().unwrap();
        let loaded = load_report(session.sink().bundle().run_dir()).unwrap();
        assert!(
            !loaded.events.is_empty(),
            "scheduler execution should emit events"
        );

        for event in &loaded.events {
            assert!(
                event.attrs.contains_key("swarmtorch.node_key"),
                "scheduler event should contain swarmtorch.node_key"
            );
            assert!(
                !event.attrs.contains_key("node_key"),
                "legacy node_key key must not be emitted"
            );
            assert!(
                !event.attrs.contains_key("detail"),
                "legacy detail key must not be emitted"
            );
        }

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scheduler_artifacts_use_swarmtorch_namespace_end_to_end() {
        let (base, mut session) = create_session("namespace_end_to_end");
        register_source(&mut session, "dataset://ns/raw");

        let node = make_node(
            "node/end_to_end",
            "passthrough",
            &["dataset://ns/raw"],
            &["dataset://ns/out"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("namespace-end-to-end".to_string()),
            nodes: vec![node],
            edges: vec![],
        };

        let report = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x66; 16]),
            test_clock,
        )
        .expect("scheduler run should succeed");
        assert_eq!(report.executed_nodes, vec!["node/end_to_end"]);

        session.finalize().expect("session finalize should succeed");
        let loaded = load_report(session.sink().bundle().run_dir()).expect("report should load");

        assert!(
            !loaded.events.is_empty(),
            "scheduler should emit namespaced events"
        );
        assert!(
            !loaded.spans.is_empty(),
            "native runner should emit namespaced spans"
        );

        for event in &loaded.events {
            assert!(event.attrs.contains_key("swarmtorch.node_key"));
            assert!(!event.attrs.contains_key("node_key"));
            assert!(!event.attrs.contains_key("detail"));
        }
        for span in &loaded.spans {
            assert!(span.attrs.contains_key("swarmtorch.op_type"));
            assert!(span.attrs.contains_key("swarmtorch.node_key"));
            assert!(span.attrs.contains_key("swarmtorch.input_count"));
            assert!(span.attrs.contains_key("swarmtorch.output_count"));
            assert!(!span.attrs.contains_key("op_type"));
            assert!(!span.attrs.contains_key("node_key"));
            assert!(!span.attrs.contains_key("input_count"));
            assert!(!span.attrs.contains_key("output_count"));
        }

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scheduler_enforces_wasm_sandbox_isolation_for_sandboxed_extension_nodes() {
        use crate::wasm_sandbox::{SandboxedExtensionPolicyV1, WASM_MAGIC_AND_VERSION_V1};
        use std::collections::BTreeMap;

        // Build minimal real .wasm module exporting transform(ptr=0, len) -> len that XORs mem[0..len] with 0x20
        let ops: Vec<u8> = vec![
            0x41, 0x00, 0x21, 0x02, 0x03, 0x40, 0x20, 0x02, 0x20, 0x01, 0x48, 0x04, 0x40, 0x20,
            0x00, 0x20, 0x02, 0x6a, 0x20, 0x00, 0x20, 0x02, 0x6a, 0x2d, 0x00, 0x00, 0x41, 0x20,
            0x73, 0x3a, 0x00, 0x00, 0x20, 0x02, 0x41, 0x01, 0x6a, 0x21, 0x02, 0x0c, 0x01, 0x0b,
            0x0b, 0x20, 0x01,
        ];
        let mut wasm = WASM_MAGIC_AND_VERSION_V1.to_vec();
        // sec 1: type (i32, i32) -> i32
        wasm.extend_from_slice(&[1, 7, 1, 0x60, 2, 0x7f, 0x7f, 1, 0x7f]);
        // sec 3: func 0 -> type 0
        wasm.extend_from_slice(&[3, 2, 1, 0]);
        // sec 5: memory init 1 max 1
        wasm.extend_from_slice(&[5, 4, 1, 0x01, 1, 1]);
        // sec 7: export "transform" -> func 0
        wasm.extend_from_slice(&[7, 13, 1, 9]);
        wasm.extend_from_slice(b"transform");
        wasm.extend_from_slice(&[0x00, 0x00]);
        // sec 10: 1 local i32 + ops + end
        let mut body = vec![1, 1, 0x7f];
        body.extend_from_slice(&ops);
        body.push(0x0b);
        wasm.push(10);
        wasm.push((body.len() + 2) as u8);
        wasm.push(1);
        wasm.push(body.len() as u8);
        wasm.extend_from_slice(&body);

        let hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let hex_hash: String = hash.iter().map(|b| format!("{b:02x}")).collect();

        let (base, mut session) = create_session("wasm_isolation");
        register_source(&mut session, "dataset://ns/raw");

        let mut wasm_node = make_node(
            "node/wasm_op",
            "wasm_transform",
            &["dataset://ns/raw"],
            &["dataset://ns/wasm_out"],
            ExecutionTrust::SandboxedExtension,
        );
        wasm_node.params.insert(
            "module_sha256".to_string(),
            swarm_torch_core::run_graph::CanonValue::Str(hex_hash),
        );
        wasm_node.params.insert(
            "export".to_string(),
            swarm_torch_core::run_graph::CanonValue::Str("transform".to_string()),
        );

        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("wasm-isolation-test".to_string()),
            nodes: vec![wasm_node],
            edges: vec![],
        };

        // 1. Under InProcessTrusted, SandboxedExtension MUST fail closed!
        let r_denied = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            RunId::from_bytes([0x67; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_denied.failed_nodes, vec!["node/wasm_op"]);

        // 2. Under WasmSandboxedV1 with allowlisted module, SandboxedExtension executes cleanly!
        let wasm_policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash);
        let mut registry = BTreeMap::new();
        registry.insert(hash, wasm);
        let r_ok = execute_graph_with_isolation(
            &graph,
            &mut session,
            &NativeOpRunner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy),
            Some(&registry),
            RunId::from_bytes([0x68; 16]),
            test_clock,
        )
        .unwrap();
        assert!(r_ok.failed_nodes.is_empty());
        assert_eq!(r_ok.executed_nodes, vec!["node/wasm_op"]);

        let _ = fs::remove_dir_all(&base);
    }
}
