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
use crate::native_runner::{ExecutionContext, NativeOpRunner, RunnerExecutionModeV1};

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

/// Execution isolation profile for graph node runners (`ADR-0023` §3, `ADR-0025`, `ST-09`).
///
/// - [`RunnerIsolationProfile::InProcessTrusted`] executes `ExecutionTrust::Core` nodes in-process.
/// - [`RunnerIsolationProfile::WasmSandboxedV1`] executes `ExecutionTrust::SandboxedExtension`
///   `wasm_transform` nodes strictly through a configured external `WasmExtensionExecutor`
///   (such as `adapters/swarm-torch-wasm` `wasmi`, currently scoped to `raw_mem0_v0`) and
///   fails closed if none is attached.
/// - [`RunnerIsolationProfile::ExternalSandboxedStub`] remains a fail-closed placeholder stub for
///   future OS-level external subprocess isolation of `UnsafeExtension` nodes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum RunnerIsolationProfile {
    /// Execute trusted core operators in-process (`ExecutionTrust::Core`).
    #[default]
    InProcessTrusted,
    /// Fail-closed isolated `wasmi` extension execution (`ExecutionTrust::SandboxedExtension`),
    /// delegating exclusively to the attached [`crate::wasm_sandbox::WasmExtensionExecutor`]
    /// (`adapters/swarm-torch-wasm`, currently scoped to `raw_mem0_v0`).
    WasmSandboxedV1(crate::wasm_sandbox::SandboxedExtensionPolicyV1),
    /// Placeholder stub for external process isolation (always denied fail-closed).
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
                record_failed_scheduler_node(
                    session,
                    trace_id,
                    &node,
                    started,
                    0,
                    "scheduler/node_denied",
                    error_code,
                    &mut report.failed_nodes,
                )?;
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
            session.invalidate_node_outputs(&node)?;
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

        // Fail-closed isolation enforcement BEFORE cache-hit evaluation so a previously cached
        // extension node cannot bypass stricter runtime isolation constraints.
        let wasm_arg = match (&node.execution_trust, isolation) {
            (_, RunnerIsolationProfile::ExternalSandboxedStub { .. }) => {
                record_failed_scheduler_node(
                    session,
                    trace_id,
                    &node,
                    started,
                    0,
                    "scheduler/node_denied",
                    "isolation_denied:ExternalSandboxedStub has no permitted host executor"
                        .to_string(),
                    &mut report.failed_nodes,
                )?;
                continue;
            }
            (swarm_torch_core::run_graph::ExecutionTrust::Core, _) => {
                if node.op_type == "wasm_transform" {
                    record_failed_scheduler_node(
                        session,
                        trace_id,
                        &node,
                        started,
                        0,
                        "scheduler/node_denied",
                        "isolation_denied:wasm_transform requires SandboxedExtension trust"
                            .to_string(),
                        &mut report.failed_nodes,
                    )?;
                    continue;
                }
                None
            }
            (
                swarm_torch_core::run_graph::ExecutionTrust::SandboxedExtension,
                RunnerIsolationProfile::WasmSandboxedV1(wasm_policy),
            ) => {
                if node.op_type != "wasm_transform" {
                    record_failed_scheduler_node(
                        session,
                        trace_id,
                        &node,
                        started,
                        0,
                        "scheduler/node_denied",
                        format!(
                            "isolation_denied:SandboxedExtension only permits op_type 'wasm_transform', got '{}'",
                            node.op_type
                        ),
                        &mut report.failed_nodes,
                    )?;
                    continue;
                }
                if runner.wasm_executor().is_none() {
                    record_failed_scheduler_node(
                        session,
                        trace_id,
                        &node,
                        started,
                        0,
                        "scheduler/node_denied",
                        format!(
                            "isolation_denied:{}",
                            crate::wasm_sandbox::WasmSandboxError::UnsupportedUntrustedWasmSandbox
                        ),
                        &mut report.failed_nodes,
                    )?;
                    continue;
                }
                let Some(registry) = wasm_registry.filter(|reg| !reg.is_empty()) else {
                    record_failed_scheduler_node(
                        session,
                        trace_id,
                        &node,
                        started,
                        0,
                        "scheduler/node_denied",
                        "isolation_denied:SandboxedExtension requires non-empty wasm_registry"
                            .to_string(),
                        &mut report.failed_nodes,
                    )?;
                    continue;
                };
                if let Err(error) = crate::wasm_sandbox::resolve_and_preflight_wasm_node_v1(
                    wasm_policy,
                    &node.params,
                    registry,
                ) {
                    record_failed_scheduler_node(
                        session,
                        trace_id,
                        &node,
                        started,
                        0,
                        "scheduler/node_denied",
                        format!("isolation_denied:{error}"),
                        &mut report.failed_nodes,
                    )?;
                    continue;
                }
                Some((wasm_policy, registry))
            }
            (swarm_torch_core::run_graph::ExecutionTrust::SandboxedExtension, _) => {
                record_failed_scheduler_node(
                    session,
                    trace_id,
                    &node,
                    started,
                    0,
                    "scheduler/node_denied",
                    "isolation_denied:SandboxedExtension requires RunnerIsolationProfile::WasmSandboxedV1"
                        .to_string(),
                    &mut report.failed_nodes,
                )?;
                continue;
            }
            (swarm_torch_core::run_graph::ExecutionTrust::UnsafeExtension, _) => {
                record_failed_scheduler_node(
                    session,
                    trace_id,
                    &node,
                    started,
                    0,
                    "scheduler/node_denied",
                    "isolation_denied:UnsafeExtension has no permitted host executor".to_string(),
                    &mut report.failed_nodes,
                )?;
                continue;
            }
        };
        let execution_profile_override = wasm_arg.map(|(wasm_policy, _)| {
            format!(
                "sandboxed_extension:{}",
                wasm_policy.policy_fingerprint_hex()
            )
        });

        let mut core_specs: Vec<OutputSpecCore> = Vec::with_capacity(node.outputs.len());
        let mut output_specs: Vec<OutputSpec> = Vec::with_capacity(node.outputs.len());
        for output in &node.outputs {
            let cached_schema = session
                .cached_output_spec(&output.asset_key)
                .and_then(|c| c.schema);
            core_specs.push(OutputSpecCore {
                asset_key: output.asset_key.clone(),
                schema: cached_schema,
            });
            // Keep fresh-execution output_specs schema unset until cache hit or runner outcome.
            output_specs.push(OutputSpec {
                asset_key: output.asset_key.clone(),
                schema: None,
                rows: None,
                bytes: None,
            });
        }

        let cache_decision = match session.verify_cache_hit_with_profile(
            &node,
            &core_specs,
            execution_profile_override.as_deref(),
        ) {
            Ok(decision) => decision,
            Err(error) => {
                let finished = (clock_nanos)();
                let error_code = format!("cache_integrity_error:{error}");
                record_failed_scheduler_node(
                    session,
                    trace_id,
                    &node,
                    finished,
                    0,
                    "scheduler/node_failed",
                    error_code,
                    &mut report.failed_nodes,
                )?;
                continue;
            }
        };
        if cache_decision == CacheDecisionV0::Hit {
            for spec in &mut output_specs {
                if let Some(cached) = session.cached_output_spec(&spec.asset_key) {
                    if spec.schema.is_none() {
                        spec.schema = cached.schema;
                    }
                    if spec.rows.is_none() {
                        spec.rows = cached.rows;
                    }
                    if spec.bytes.is_none() {
                        spec.bytes = cached.bytes;
                    }
                }
            }
            let finished = (clock_nanos)();
            let empty_cas = BTreeMap::new();
            if let Err(error) = session.materialize_node_outputs_with_cas_bindings_and_profile(
                &node,
                &output_specs,
                &empty_cas,
                finished,
                CacheDecisionV0::Hit,
                0,
                execution_profile_override.as_deref(),
            ) {
                let error_code = format!("materialization_error:{error}");
                record_failed_scheduler_node(
                    session,
                    trace_id,
                    &node,
                    finished,
                    0,
                    "scheduler/node_failed",
                    error_code,
                    &mut report.failed_nodes,
                )?;
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

        let mut input_content_types = BTreeMap::new();
        for input in &node.inputs {
            if let Some(ct) = session.asset_content_type(&input.asset_key) {
                input_content_types.insert(input.asset_key.clone(), ct.to_string());
            }
        }
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let runner_result = runner.run_in_session_with_content_types(
            &context,
            &run_dir,
            &node,
            &inputs,
            session.sink().as_ref(),
            wasm_arg,
            &input_content_types,
        );
        let finished = (clock_nanos)();
        let duration_ms = finished.saturating_sub(started) / 1_000_000;
        let exec_outcome = match runner_result {
            Ok(outcome) => outcome,
            Err(error) => {
                record_failed_scheduler_node(
                    session,
                    trace_id,
                    &node,
                    finished,
                    duration_ms,
                    "scheduler/node_failed",
                    format!("runner_error:{error}"),
                    &mut report.failed_nodes,
                )?;
                continue;
            }
        };
        for phys_spec in &exec_outcome.output_specs {
            if let Some(target_spec) = output_specs
                .iter_mut()
                .find(|s| s.asset_key == phys_spec.asset_key)
            {
                if phys_spec.schema.is_some() {
                    target_spec.schema = phys_spec.schema.clone();
                }
                if phys_spec.rows.is_some() {
                    target_spec.rows = phys_spec.rows;
                }
                if phys_spec.bytes.is_some() {
                    target_spec.bytes = phys_spec.bytes;
                }
            }
        }
        let runner_instances = exec_outcome.instances;
        let output_content_types = exec_outcome.output_content_types;
        let physical_cas_required = runner.mode() == RunnerExecutionModeV1::PhysicalStrict
            || node.op_type == "wasm_transform"
            || (!inputs.is_empty()
                && inputs
                    .iter()
                    .all(|i| i.uri.as_deref().and_then(parse_cas_uri_digest).is_some()));
        let cas_bindings = if physical_cas_required {
            match build_physical_cas_bindings(&node, &runner_instances, &output_content_types) {
                Ok(bindings) => bindings,
                Err(error) => {
                    record_failed_scheduler_node(
                        session,
                        trace_id,
                        &node,
                        finished,
                        duration_ms,
                        "scheduler/node_failed",
                        format!("materialization_error:{error}"),
                        &mut report.failed_nodes,
                    )?;
                    continue;
                }
            }
        } else {
            BTreeMap::new()
        };

        let materialize = session.materialize_node_outputs_with_cas_bindings_and_profile(
            &node,
            &output_specs,
            &cas_bindings,
            finished,
            cache_decision,
            duration_ms,
            execution_profile_override.as_deref(),
        );
        if let Err(error) = materialize {
            record_failed_scheduler_node(
                session,
                trace_id,
                &node,
                finished,
                duration_ms,
                "scheduler/node_failed",
                format!("materialization_error:{error}"),
                &mut report.failed_nodes,
            )?;
            continue;
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

#[allow(clippy::too_many_arguments)]
fn record_failed_scheduler_node(
    session: &mut DataOpsSession,
    trace_id: TraceId,
    node: &NodeV1,
    ts_unix_nanos: u64,
    duration_ms: u64,
    event_name: &str,
    error_code: String,
    failed_nodes: &mut Vec<String>,
) -> io::Result<()> {
    session.invalidate_node_outputs(node)?;
    session.record_node_error(node, ts_unix_nanos, duration_ms, error_code.clone())?;
    emit_scheduler_event(
        session.sink().as_ref(),
        trace_id,
        ts_unix_nanos,
        event_name,
        &node.node_key,
        Some(&error_code),
    )?;
    failed_nodes.push(node.node_key.clone());
    Ok(())
}

fn build_physical_cas_bindings(
    node: &NodeV1,
    runner_instances: &[AssetInstanceV1],
    output_content_types: &BTreeMap<String, String>,
) -> io::Result<BTreeMap<String, (String, String)>> {
    let mut declared = BTreeSet::new();
    for out in &node.outputs {
        if !declared.insert(out.asset_key.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "duplicate declared output asset_key '{}' in node '{}'",
                    out.asset_key, node.node_key
                ),
            ));
        }
    }

    let mut cas_bindings = BTreeMap::new();
    for instance in runner_instances {
        if !declared.contains(instance.asset_key.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "runner emitted undeclared output asset_key '{}' for node '{}'",
                    instance.asset_key, node.node_key
                ),
            ));
        }
        let uri = instance.uri.as_deref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "physical runner output '{}' missing CAS URI",
                    instance.asset_key
                ),
            )
        })?;
        if !uri.starts_with("cas://sha256/") || parse_cas_uri_digest(uri).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "physical runner output '{}' has invalid CAS URI '{uri}'",
                    instance.asset_key
                ),
            ));
        }
        let content_type = output_content_types
            .get(&instance.asset_key)
            .filter(|ct| !ct.is_empty())
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "physical runner output '{}' missing content_type",
                        instance.asset_key
                    ),
                )
            })?;
        if cas_bindings
            .insert(instance.asset_key.clone(), (uri.to_string(), content_type))
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "runner emitted duplicate output asset_key '{}' for node '{}'",
                    instance.asset_key, node.node_key
                ),
            ));
        }
    }

    if cas_bindings.len() != declared.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "physical runner produced {} CAS outputs for node '{}' declaring {} outputs",
                cas_bindings.len(),
                node.node_key,
                declared.len()
            ),
        ));
    }

    Ok(cas_bindings)
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
    use crate::native_runner::{
        ColumnDtypeV1, ColumnSchemaV1, ColumnarTableV1, RunnerExecutionModeV1,
        CONTENT_TYPE_COLUMNAR_JSON_V1, CONTENT_TYPE_NDJSON,
    };
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
            &NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests),
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

        let test_runner = NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests);

        // First run: CacheDecisionV0::Miss
        let r1 = execute_graph_sequential(
            &graph,
            &mut session,
            &test_runner,
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
            &test_runner,
            &PermissivePolicy,
            RunId::from_bytes([0x12; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r2.cached_nodes, vec!["node/cacheable"]);

        // Delete the physical CAS blob on disk -> explicit integrity failure, NEVER silent Miss!
        let blob_path = session
            .sink()
            .bundle()
            .run_dir()
            .join("blobs")
            .join("sha256")
            .join(digest.to_hex());
        fs::remove_file(&blob_path).unwrap();

        let direct_err = session
            .verify_cache_hit(
                &node,
                &[OutputSpecCore {
                    asset_key: "dataset://ns/cached_out".to_string(),
                    schema: None,
                }],
            )
            .unwrap_err();
        assert_eq!(direct_err.kind(), io::ErrorKind::NotFound);

        let r3 = execute_graph_sequential(
            &graph,
            &mut session,
            &test_runner,
            &PermissivePolicy,
            RunId::from_bytes([0x12; 16]),
            test_clock,
        )
        .unwrap();
        assert!(r3.cached_nodes.is_empty());
        assert!(
            r3.executed_nodes.is_empty(),
            "deleting physical CAS blob must fail closed rather than silently re-executing"
        );
        assert_eq!(r3.failed_nodes, vec!["node/cacheable"]);

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
            &test_runner,
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
            loaded.materializations[2].status,
            MaterializationStatusV0::Error
        );
        assert!(loaded.materializations[2]
            .error_code
            .as_deref()
            .is_some_and(|code| code.starts_with("cache_integrity_error:")));
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
            &NativeOpRunner::new(),
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
            &NativeOpRunner::new(),
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
            &NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests),
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
            &NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests),
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
            &NativeOpRunner::new(),
            &PermissivePolicy,
            RunId::from_bytes([0x67; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_denied.failed_nodes, vec!["node/wasm_op"]);

        // 2. Under WasmSandboxedV1 without an attached WasmExtensionExecutor, untrusted Wasm MUST fail closed cleanly!
        let wasm_policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash);
        let mut registry = BTreeMap::new();
        registry.insert(hash, wasm);
        let r_blocked = execute_graph_with_isolation(
            &graph,
            &mut session,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy),
            Some(&registry),
            RunId::from_bytes([0x68; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_blocked.failed_nodes, vec!["node/wasm_op"]);
        assert!(r_blocked.executed_nodes.is_empty());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st10_scheduler_physical_strict_dag_propagates_formats_metrics_and_reopen_cache_hits() {
        use sha2::{Digest, Sha256};
        use swarm_torch_core::run_graph::CanonValue;

        use crate::artifacts::{
            hex_lower, write_once_cas_blob, ArtifactWriteProfile, LockAcquisitionPolicy,
        };

        let base = temp_dir("st10_physical_dag");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([0x7A; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();

        let cols = vec![
            ColumnSchemaV1 {
                name: "id".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            },
            ColumnSchemaV1 {
                name: "score".to_string(),
                dtype: ColumnDtypeV1::F64,
                nullable: false,
            },
        ];
        let tbl1 = ColumnarTableV1::try_new(
            cols.clone(),
            vec![
                vec![serde_json::Value::from(1), serde_json::Value::from(0.9)],
                vec![serde_json::Value::from(2), serde_json::Value::from(0.2)],
                vec![serde_json::Value::from(3), serde_json::Value::from(0.85)],
            ],
        )
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();
        let tbl2 = ColumnarTableV1::try_new(
            cols,
            vec![vec![
                serde_json::Value::from(4),
                serde_json::Value::from(0.95),
            ]],
        )
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();

        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let sha1: [u8; 32] = Sha256::digest(&tbl1).into();
        let sha2: [u8; 32] = Sha256::digest(&tbl2).into();
        write_once_cas_blob(&run_dir, &sha1, &tbl1).unwrap();
        write_once_cas_blob(&run_dir, &sha2, &tbl2).unwrap();

        let ingest = make_node("ingest/shards", "ingest", &[], &[], ExecutionTrust::Core);
        session
            .register_source(
                "dataset://ns/shard1",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{}", hex_lower(&sha1)),
                    content_type: CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
                    auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();
        session
            .register_source(
                "dataset://ns/shard2",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{}", hex_lower(&sha2)),
                    content_type: CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
                    auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut filter_node = make_node(
            "node/filter",
            "filter_rows",
            &["dataset://ns/shard1"],
            &["dataset://ns/filtered"],
            ExecutionTrust::Core,
        );
        filter_node
            .params
            .insert("column".to_string(), CanonValue::Str("score".to_string()));
        filter_node
            .params
            .insert("op".to_string(), CanonValue::Str("ge".to_string()));
        filter_node
            .params
            .insert("value".to_string(), CanonValue::Str("0.8".to_string()));
        let union_node = make_node(
            "node/union",
            "union",
            &["dataset://ns/filtered", "dataset://ns/shard2"],
            &["dataset://ns/merged"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("st10-physical".to_string()),
            nodes: vec![filter_node.clone(), union_node.clone()],
            edges: vec![EdgeV1 {
                from_node_id: node_id_from_key(&filter_node.node_key),
                to_node_id: node_id_from_key(&union_node.node_key),
                asset_key: Some("dataset://ns/filtered".to_string()),
            }],
        };

        // Run 1 with default PhysicalStrict NativeOpRunner::new(): filter -> union end-to-end!
        let r1 = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert!(r1.failed_nodes.is_empty(), "{:?}", r1.failed_nodes);
        assert_eq!(r1.executed_nodes, vec!["node/filter", "node/union"]);
        assert!(r1.cached_nodes.is_empty());
        assert_eq!(
            session.asset_content_type("dataset://ns/filtered"),
            Some(CONTENT_TYPE_COLUMNAR_JSON_V1)
        );
        assert_eq!(
            session.asset_content_type("dataset://ns/merged"),
            Some(CONTENT_TYPE_COLUMNAR_JSON_V1)
        );

        session.finalize().unwrap();
        let r1_merged = load_report(&run_dir)
            .unwrap()
            .materializations
            .into_iter()
            .find(|m| m.asset_key == "dataset://ns/merged")
            .unwrap();
        assert_eq!(r1_merged.rows, Some(3));
        assert!(r1_merged.bytes.unwrap() > 0);

        // Close and reopen session -> Run 2 must hit cache and preserve CAS URI, content_type, rows, bytes!
        drop(session);
        let mut reopened = DataOpsSession::open(
            &run_dir,
            ArtifactWriteProfile::default(),
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let r2 = execute_graph_sequential(
            &graph,
            &mut reopened,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert!(r2.failed_nodes.is_empty());
        assert_eq!(r2.cached_nodes, vec!["node/filter", "node/union"]);
        assert_eq!(
            reopened.asset_content_type("dataset://ns/merged"),
            Some(CONTENT_TYPE_COLUMNAR_JSON_V1)
        );

        reopened.finalize().unwrap();
        let hit_merged = load_report(&run_dir)
            .unwrap()
            .materializations
            .into_iter()
            .rfind(|m| m.asset_key == "dataset://ns/merged")
            .unwrap();
        assert_eq!(hit_merged.cache_decision, CacheDecisionV0::Hit);
        assert_eq!(hit_merged.rows, r1_merged.rows);
        assert_eq!(hit_merged.bytes, r1_merged.bytes);

        // Also test NDJSON DAG (`{...}\n` rows) end-to-end through PhysicalStrict scheduler
        let nd_bytes = b"{\"id\":10,\"score\":0.9}\n{\"id\":11,\"score\":0.1}\n";
        let sha_nd: [u8; 32] = Sha256::digest(nd_bytes).into();
        write_once_cas_blob(&run_dir, &sha_nd, nd_bytes).unwrap();
        reopened
            .register_source(
                "dataset://ns/nd_raw",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{}", hex_lower(&sha_nd)),
                    content_type: CONTENT_TYPE_NDJSON.to_string(),
                    auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();
        let mut nd_filter = make_node(
            "node/nd_filter",
            "filter_rows",
            &["dataset://ns/nd_raw"],
            &["dataset://ns/nd_filtered"],
            ExecutionTrust::Core,
        );
        nd_filter
            .params
            .insert("column".to_string(), CanonValue::Str("score".to_string()));
        nd_filter
            .params
            .insert("op".to_string(), CanonValue::Str("ge".to_string()));
        nd_filter
            .params
            .insert("value".to_string(), CanonValue::Str("0.5".to_string()));
        let nd_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("st10-ndjson".to_string()),
            nodes: vec![nd_filter],
            edges: vec![],
        };
        let r_nd = execute_graph_sequential(
            &nd_graph,
            &mut reopened,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert!(r_nd.failed_nodes.is_empty());
        assert_eq!(
            reopened.asset_content_type("dataset://ns/nd_filtered"),
            Some(CONTENT_TYPE_NDJSON)
        );
        reopened.finalize().unwrap();
        let nd_mat = load_report(&run_dir)
            .unwrap()
            .materializations
            .into_iter()
            .rfind(|m| m.asset_key == "dataset://ns/nd_filtered")
            .unwrap();
        assert_eq!(nd_mat.rows, Some(1));

        drop(reopened);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st15_scheduler_invalidates_stale_outputs_on_failure_and_binds_cas_atomically() {
        use sha2::{Digest, Sha256};
        use swarm_torch_core::execution::ExecutionPolicy;
        use swarm_torch_core::run_graph::{CanonValue, EdgeV1};

        use crate::artifacts::{hex_lower, parse_cas_uri_digest, write_once_cas_blob};
        use crate::native_runner::{ColumnDtypeV1, ColumnSchemaV1, ColumnarTableV1};

        struct CustomPolicy {
            deny_produce: bool,
        }
        impl ExecutionPolicy for CustomPolicy {
            fn allow(
                &self,
                node: &NodeV1,
                _registry: &swarm_torch_core::dataops::DatasetRegistryV1,
            ) -> PolicyDecision {
                if self.deny_produce && node.node_key == "node/produce" {
                    PolicyDecision::Denied {
                        reason: "policy_revoked".to_string(),
                    }
                } else {
                    PolicyDecision::Allowed
                }
            }
        }

        let (base, mut session) = create_session("st15_sched_invalidation_atomic_cas");
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let tbl = ColumnarTableV1::try_new(
            vec![ColumnSchemaV1 {
                name: "score".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            }],
            vec![
                vec![serde_json::Value::from(10)],
                vec![serde_json::Value::from(3)],
            ],
        )
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();
        let raw_sha: [u8; 32] = Sha256::digest(&tbl).into();
        write_once_cas_blob(&run_dir, &raw_sha, &tbl).unwrap();
        let ingest = make_node("ingest/raw", "ingest", &[], &[], ExecutionTrust::Core);
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{}", hex_lower(&raw_sha)),
                    content_type: CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
                    auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                    etag_or_version: Some("v1".to_string()),
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut produce = make_node(
            "node/produce",
            "filter_rows",
            &["dataset://ns/raw"],
            &["dataset://ns/prod_out"],
            ExecutionTrust::Core,
        );
        produce
            .params
            .insert("column".to_string(), CanonValue::Str("score".to_string()));
        produce
            .params
            .insert("op".to_string(), CanonValue::Str("ge".to_string()));
        produce
            .params
            .insert("value".to_string(), CanonValue::Str("5".to_string()));
        let downstream = make_node(
            "node/downstream",
            "passthrough",
            &["dataset://ns/prod_out"],
            &["dataset://ns/final"],
            ExecutionTrust::Core,
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("st15b1".to_string()),
            nodes: vec![produce.clone(), downstream.clone()],
            edges: vec![EdgeV1 {
                from_node_id: node_id_from_key(&produce.node_key),
                to_node_id: node_id_from_key(&downstream.node_key),
                asset_key: None,
            }],
        };

        let runner = NativeOpRunner::new();

        // 1. Valid execution binds CAS URI atomically with the first registry snapshot/update.
        let r1 = execute_graph_sequential(
            &graph,
            &mut session,
            &runner,
            &CustomPolicy {
                deny_produce: false,
            },
            RunId::from_bytes([0xE5; 16]),
            test_clock,
        )
        .unwrap();
        assert!(r1.failed_nodes.is_empty(), "{:?}", r1.failed_nodes);
        assert!(session.fingerprint("dataset://ns/prod_out").is_some());
        assert!(session.fingerprint("dataset://ns/final").is_some());

        let prod_entry = session
            .registry_snapshot()
            .datasets
            .into_iter()
            .find(|d| d.asset_key == "dataset://ns/prod_out")
            .unwrap();
        let prod_cas_uri = prod_entry
            .source
            .as_ref()
            .expect("bound CAS source")
            .uri
            .clone();
        assert!(prod_cas_uri.starts_with("cas://sha256/"));
        assert_eq!(
            session.asset_content_type("dataset://ns/prod_out"),
            Some(CONTENT_TYPE_COLUMNAR_JSON_V1)
        );
        let reg_updates =
            fs::read_to_string(run_dir.join("datasets/registry_updates.ndjson")).unwrap();
        let prod_update_lines: Vec<&str> = reg_updates
            .lines()
            .filter(|line| line.contains("\"dataset://ns/prod_out\""))
            .collect();
        assert_eq!(
            prod_update_lines.len(),
            1,
            "atomic materialization must emit a single already-CAS-bound registry_updates row: {prod_update_lines:?}"
        );
        assert!(
            prod_update_lines[0].contains(&prod_cas_uri),
            "registry update row must carry CAS URI atomically"
        );

        // 2. Policy denial invalidates stale `dataset://ns/prod_out` and skips/invalidates `dataset://ns/final`.
        let r2 = execute_graph_sequential(
            &graph,
            &mut session,
            &runner,
            &CustomPolicy { deny_produce: true },
            RunId::from_bytes([0xE5; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r2.failed_nodes, vec!["node/produce"]);
        assert_eq!(r2.skipped_nodes, vec!["node/downstream"]);
        assert!(session.fingerprint("dataset://ns/prod_out").is_none());
        assert!(session
            .resolve_asset_instance("dataset://ns/prod_out")
            .is_none());
        assert!(session.fingerprint("dataset://ns/final").is_none());

        // 3. Re-populate cleanly, then trigger a runner error -> invalidates stale outputs before continuing.
        let r3_ok = execute_graph_sequential(
            &graph,
            &mut session,
            &runner,
            &CustomPolicy {
                deny_produce: false,
            },
            RunId::from_bytes([0xE5; 16]),
            test_clock,
        )
        .unwrap();
        assert!(r3_ok.failed_nodes.is_empty());
        assert!(session.fingerprint("dataset://ns/prod_out").is_some());
        assert!(session.fingerprint("dataset://ns/final").is_some());

        let mut broken_produce = produce.clone();
        broken_produce.params.insert(
            "column".to_string(),
            CanonValue::Str("missing_col".to_string()),
        );
        broken_produce.cache_policy = Some(CachePolicyV0::Bypass);
        let broken_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("st15b1-broken".to_string()),
            nodes: vec![broken_produce.clone(), downstream.clone()],
            edges: vec![EdgeV1 {
                from_node_id: node_id_from_key(&broken_produce.node_key),
                to_node_id: node_id_from_key(&downstream.node_key),
                asset_key: None,
            }],
        };
        let r4_fail = execute_graph_sequential(
            &broken_graph,
            &mut session,
            &runner,
            &CustomPolicy {
                deny_produce: false,
            },
            RunId::from_bytes([0xE5; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r4_fail.failed_nodes, vec!["node/produce"]);
        assert_eq!(r4_fail.skipped_nodes, vec!["node/downstream"]);
        assert!(session.fingerprint("dataset://ns/prod_out").is_none());
        assert!(session.fingerprint("dataset://ns/final").is_none());

        // 4. Re-populate cleanly, then corrupt/delete cached CAS blob -> verify_cache_hit error invalidates stale outputs!
        let r5_ok = execute_graph_sequential(
            &graph,
            &mut session,
            &runner,
            &CustomPolicy {
                deny_produce: false,
            },
            RunId::from_bytes([0xE5; 16]),
            test_clock,
        )
        .unwrap();
        assert!(r5_ok.failed_nodes.is_empty());
        let prod_entry2 = session
            .registry_snapshot()
            .datasets
            .into_iter()
            .find(|d| d.asset_key == "dataset://ns/prod_out")
            .unwrap();
        let prod_digest = parse_cas_uri_digest(&prod_entry2.source.unwrap().uri).unwrap();
        let blob_path = run_dir
            .join("blobs")
            .join("sha256")
            .join(hex_lower(&prod_digest));
        fs::remove_file(blob_path).unwrap();

        let r6_cache_err = execute_graph_sequential(
            &graph,
            &mut session,
            &runner,
            &CustomPolicy {
                deny_produce: false,
            },
            RunId::from_bytes([0xE5; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r6_cache_err.failed_nodes, vec!["node/produce"]);
        assert_eq!(r6_cache_err.skipped_nodes, vec!["node/downstream"]);
        assert!(session.fingerprint("dataset://ns/prod_out").is_none());
        assert!(session.fingerprint("dataset://ns/final").is_none());

        drop(session);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st15_isolation_runs_before_cache_hit_and_external_stub_fails_closed() {
        use sha2::{Digest, Sha256};
        use swarm_torch_core::run_graph::CanonValue;

        use crate::artifacts::{hex_lower, write_once_cas_blob};
        use crate::native_runner::CONTENT_TYPE_OCTET_STREAM;
        use crate::wasm_sandbox::{
            SandboxedExtensionPolicyV1, WasmExecutionReceiptV1, WasmExtensionExecutor,
            WasmSandboxError, WASM_MAGIC_AND_VERSION_V1,
        };

        struct StubWasmExecutor;
        impl WasmExtensionExecutor for StubWasmExecutor {
            fn execute_byte_transform(
                &self,
                policy: &SandboxedExtensionPolicyV1,
                wasm_bytes: &[u8],
                _export_name: &str,
                input_payload: &[u8],
            ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
                Ok(WasmExecutionReceiptV1 {
                    module_sha256: SandboxedExtensionPolicyV1::sha256_module(wasm_bytes),
                    fuel_consumed: policy.limits.max_fuel.min(100),
                    peak_memory_bytes: 65_536,
                    return_code: 0,
                    output_bytes: input_payload.to_vec(),
                })
            }

            fn execute_scalar_i32(
                &self,
                _policy: &SandboxedExtensionPolicyV1,
                _wasm_bytes: &[u8],
                _export_name: &str,
                _arg: i32,
            ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
                unreachable!()
            }
        }

        // Valid minimal module: (i32, i32) -> i32 export "transform" returning 0
        let mut wasm = WASM_MAGIC_AND_VERSION_V1.to_vec();
        wasm.extend_from_slice(&[1, 7, 1, 0x60, 2, 0x7f, 0x7f, 1, 0x7f]);
        wasm.extend_from_slice(&[3, 2, 1, 0]);
        wasm.extend_from_slice(&[5, 4, 1, 0x01, 1, 1]);
        wasm.extend_from_slice(&[7, 13, 1, 9]);
        wasm.extend_from_slice(b"transform");
        wasm.extend_from_slice(&[0x00, 0x00]);
        wasm.extend_from_slice(&[10, 6, 1, 4, 0, 0x41, 0, 0x0b]);

        let hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let hex_hash = hex_lower(&hash);
        let mut registry = BTreeMap::new();
        registry.insert(hash, wasm);
        let wasm_policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash);

        let (base, mut session) = create_session("st15_isolation_before_cache");
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let payload = b"wasm-cas-in";
        let digest: [u8; 32] = Sha256::digest(payload).into();
        write_once_cas_blob(&run_dir, &digest, payload).unwrap();
        let ingest = make_node("ingest/wasm", "ingest", &[], &[], ExecutionTrust::Core);
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{}", hex_lower(&digest)),
                    content_type: CONTENT_TYPE_OCTET_STREAM.to_string(),
                    auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                    etag_or_version: None,
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut wasm_node = make_node(
            "node/wasm_op",
            "wasm_transform",
            &["dataset://ns/raw"],
            &["dataset://ns/wasm_out"],
            ExecutionTrust::SandboxedExtension,
        );
        wasm_node
            .params
            .insert("module_sha256".to_string(), CanonValue::Str(hex_hash));
        wasm_node.params.insert(
            "export".to_string(),
            CanonValue::Str("transform".to_string()),
        );
        let wasm_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("wasm-cache-check".to_string()),
            nodes: vec![wasm_node],
            edges: vec![],
        };

        let attached_runner = NativeOpRunner::new().with_wasm_executor(Arc::new(StubWasmExecutor));
        let r_ok = execute_graph_with_isolation(
            &wasm_graph,
            &mut session,
            &attached_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy.clone()),
            Some(&registry),
            RunId::from_bytes([0xE6; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_ok.executed_nodes, vec!["node/wasm_op"]);
        assert!(session.fingerprint("dataset://ns/wasm_out").is_some());

        // Cached entry exists, yet running under InProcessTrusted MUST deny (NOT cache hit!) and invalidate output.
        let r_inproc_denied = execute_graph_sequential(
            &wasm_graph,
            &mut session,
            &attached_runner,
            &PermissivePolicy,
            RunId::from_bytes([0xE6; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_inproc_denied.failed_nodes, vec!["node/wasm_op"]);
        assert!(r_inproc_denied.cached_nodes.is_empty());
        assert!(session.fingerprint("dataset://ns/wasm_out").is_none());

        // Re-populate cleanly, then deny under WasmSandboxedV1 without executor and without registry.
        execute_graph_with_isolation(
            &wasm_graph,
            &mut session,
            &attached_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy.clone()),
            Some(&registry),
            RunId::from_bytes([0xE6; 16]),
            test_clock,
        )
        .unwrap();
        assert!(session.fingerprint("dataset://ns/wasm_out").is_some());

        let r_no_exec = execute_graph_with_isolation(
            &wasm_graph,
            &mut session,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy.clone()),
            Some(&registry),
            RunId::from_bytes([0xE6; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_no_exec.failed_nodes, vec!["node/wasm_op"]);
        assert!(r_no_exec.cached_nodes.is_empty());
        assert!(session.fingerprint("dataset://ns/wasm_out").is_none());

        execute_graph_with_isolation(
            &wasm_graph,
            &mut session,
            &attached_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy.clone()),
            Some(&registry),
            RunId::from_bytes([0xE6; 16]),
            test_clock,
        )
        .unwrap();
        let r_no_reg = execute_graph_with_isolation(
            &wasm_graph,
            &mut session,
            &attached_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(wasm_policy),
            None,
            RunId::from_bytes([0xE6; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_no_reg.failed_nodes, vec!["node/wasm_op"]);
        assert!(r_no_reg.cached_nodes.is_empty());
        assert!(session.fingerprint("dataset://ns/wasm_out").is_none());

        // ExternalSandboxedStub MUST fail closed even for ExecutionTrust::Core nodes.
        let core_node = make_node(
            "node/core_stub",
            "passthrough",
            &["dataset://ns/raw"],
            &["dataset://ns/core_out"],
            ExecutionTrust::Core,
        );
        let core_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("stub-fail-closed".to_string()),
            nodes: vec![core_node],
            edges: vec![],
        };
        let r_stub = execute_graph_with_isolation(
            &core_graph,
            &mut session,
            &attached_runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::ExternalSandboxedStub {
                timeout_ms: 1000,
                max_memory_bytes: 1_048_576,
            },
            None,
            RunId::from_bytes([0xE6; 16]),
            test_clock,
        )
        .unwrap();
        assert_eq!(r_stub.failed_nodes, vec!["node/core_stub"]);
        assert!(r_stub.executed_nodes.is_empty());

        drop(session);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st15_schema_cache_hit_and_cas_bijection_and_reopen_invalidation() {
        use sha2::{Digest, Sha256};
        use swarm_torch_core::dataops::SchemaDescriptorV0;
        use swarm_torch_core::run_graph::CanonValue;

        use crate::artifacts::{
            hex_lower, write_once_cas_blob, ArtifactWriteProfile, LockAcquisitionPolicy,
            SnapshotProfile,
        };

        // Verify schema-bearing OutputSpec hits scheduler cache on matching run and does not leak onto Miss,
        // and verify mismatched node.outputs fails closed + invalidation survives reopen/Streaming + load_report.
        let base = temp_dir("st15_schema_bijection_reopen");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let streaming_profile = ArtifactWriteProfile {
            snapshot_profile: SnapshotProfile::streaming(8),
            ..ArtifactWriteProfile::default()
        };
        let run_id = RunId::from_bytes([0xE7; 16]);
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            streaming_profile,
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let run_dir = session.sink().bundle().run_dir().to_path_buf();

        let tbl = ColumnarTableV1::try_new(
            vec![ColumnSchemaV1 {
                name: "x".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            }],
            vec![vec![serde_json::Value::from(7)]],
        )
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();
        let sha: [u8; 32] = Sha256::digest(&tbl).into();
        write_once_cas_blob(&run_dir, &sha, &tbl).unwrap();
        let in_uri = format!("cas://sha256/{}", hex_lower(&sha));
        let ingest = make_node("ingest/x", "ingest", &[], &[], ExecutionTrust::Core);
        session
            .register_source(
                "dataset://ns/in",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: in_uri.clone(),
                    content_type: CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
                    auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                    etag_or_version: None,
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut filter_node = make_node(
            "node/filter",
            "filter_rows",
            &["dataset://ns/in"],
            &["dataset://ns/out"],
            ExecutionTrust::Core,
        );
        filter_node
            .params
            .insert("column".to_string(), CanonValue::Str("x".to_string()));
        filter_node
            .params
            .insert("op".to_string(), CanonValue::Str("eq".to_string()));
        filter_node
            .params
            .insert("value".to_string(), CanonValue::Str("7".to_string()));

        // Seed schema-bearing cached output directly, then execute matching node via scheduler -> Cache Hit!
        let mut seed_cas = BTreeMap::new();
        seed_cas.insert(
            "dataset://ns/out".to_string(),
            (in_uri, CONTENT_TYPE_COLUMNAR_JSON_V1.to_string()),
        );
        let schema_a = SchemaDescriptorV0 {
            format: "stcl-columns-v1".to_string(),
            canonical: r#"[{"name":"x","dtype":"i64","nullable":false}]"#.to_string(),
        };
        session
            .materialize_node_outputs_with_cas_bindings(
                &filter_node,
                &[OutputSpec {
                    asset_key: "dataset://ns/out".to_string(),
                    schema: Some(schema_a.clone()),
                    rows: Some(1),
                    bytes: Some(tbl.len() as u64),
                }],
                &seed_cas,
                100,
                swarm_torch_core::dataops::CacheDecisionV0::Miss,
                1,
            )
            .unwrap();

        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("schema-cache".to_string()),
            nodes: vec![filter_node.clone()],
            edges: vec![],
        };
        let r_hit = execute_graph_sequential(
            &graph,
            &mut session,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r_hit.cached_nodes, vec!["node/filter"]);
        assert_eq!(
            session
                .cached_output_spec("dataset://ns/out")
                .unwrap()
                .schema,
            Some(schema_a)
        );

        // Parameter change causes Miss -> fresh NativeOpRunner execution does not inherit stale schema_a.
        let mut miss_node = filter_node.clone();
        miss_node
            .params
            .insert("op".to_string(), CanonValue::Str("ge".to_string()));
        let miss_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("schema-miss".to_string()),
            nodes: vec![miss_node],
            edges: vec![],
        };
        let r_miss = execute_graph_sequential(
            &miss_graph,
            &mut session,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert!(r_miss.cached_nodes.is_empty());
        assert_eq!(r_miss.executed_nodes, vec!["node/filter"]);
        assert_eq!(
            session
                .cached_output_spec("dataset://ns/out")
                .unwrap()
                .schema,
            None
        );

        // Output cardinality mismatch (2 declared outputs for 1-input passthrough) MUST fail closed
        // BEFORE mutating registry AND invalidate prior `dataset://ns/out` even under Streaming!
        let bad_cardinality_node = make_node(
            "node/bad_card",
            "passthrough",
            &["dataset://ns/in"],
            &["dataset://ns/out", "dataset://ns/extra_unbound"],
            ExecutionTrust::Core,
        );
        let bad_graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("bad-card".to_string()),
            nodes: vec![bad_cardinality_node],
            edges: vec![],
        };
        let r_bad = execute_graph_sequential(
            &bad_graph,
            &mut session,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r_bad.failed_nodes, vec!["node/bad_card"]);
        assert!(session.fingerprint("dataset://ns/out").is_none());
        assert!(session.fingerprint("dataset://ns/extra_unbound").is_none());

        // Drop session WITHOUT finalize(), reopen via DataOpsSession::open -> invalidated output MUST stay gone!
        drop(session);
        let mut reopened =
            DataOpsSession::open(&run_dir, streaming_profile, LockAcquisitionPolicy::FailFast)
                .unwrap();
        assert!(reopened.fingerprint("dataset://ns/out").is_none());
        assert!(reopened.cached_output_spec("dataset://ns/out").is_none());
        let r_reopened = execute_graph_sequential(
            &miss_graph,
            &mut reopened,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert!(r_reopened.cached_nodes.is_empty());
        assert_eq!(r_reopened.executed_nodes, vec!["node/filter"]);

        // Now fail again + finalize -> load_report MUST NOT see invalidated asset in registry!
        let r_bad_after = execute_graph_sequential(
            &bad_graph,
            &mut reopened,
            &NativeOpRunner::new(),
            &PermissivePolicy,
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r_bad_after.failed_nodes, vec!["node/bad_card"]);
        reopened.finalize().unwrap();
        let report = load_report(&run_dir).unwrap();
        assert!(
            !report
                .registry
                .datasets
                .iter()
                .any(|d| d.asset_key == "dataset://ns/out"),
            "load_report must not resurrect invalidated outputs: {:?}",
            report.registry.datasets
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn st15_wasm_policy_cache_isolation_and_cas_bindings_edge_cases_and_manifest_schema() {
        use sha2::{Digest, Sha256};
        use std::sync::atomic::AtomicUsize;
        use swarm_torch_core::run_graph::CanonValue;

        use crate::artifacts::schema_check::validate_verified_bundle_schemas;
        use crate::artifacts::{
            hex_lower, write_once_cas_blob, ArtifactWriteProfile, LockAcquisitionPolicy,
            ManifestRefreshPolicy, SnapshotProfile,
        };
        use crate::native_runner::CONTENT_TYPE_OCTET_STREAM;
        use crate::wasm_sandbox::{
            SandboxedExtensionPolicyV1, WasmExecutionReceiptV1, WasmExtensionExecutor,
            WasmSandboxError, WASM_MAGIC_AND_VERSION_V1,
        };

        struct CountingWasmExecutor {
            calls: AtomicUsize,
        }
        impl WasmExtensionExecutor for CountingWasmExecutor {
            fn execute_byte_transform(
                &self,
                policy: &SandboxedExtensionPolicyV1,
                wasm_bytes: &[u8],
                _export_name: &str,
                input_payload: &[u8],
            ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(WasmExecutionReceiptV1 {
                    module_sha256: SandboxedExtensionPolicyV1::sha256_module(wasm_bytes),
                    fuel_consumed: policy.limits.max_fuel.min(50),
                    peak_memory_bytes: 65_536,
                    return_code: 0,
                    output_bytes: input_payload.to_vec(),
                })
            }

            fn execute_scalar_i32(
                &self,
                _policy: &SandboxedExtensionPolicyV1,
                _wasm_bytes: &[u8],
                _export_name: &str,
                _arg: i32,
            ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
                unreachable!()
            }
        }

        // Direct unit coverage for build_physical_cas_bindings edge cases
        let mut node_ok = make_node(
            "node/cas",
            "passthrough",
            &[],
            &["dataset://ns/a"],
            ExecutionTrust::Core,
        );
        let valid_uri = format!("cas://sha256/{}", "ab".repeat(32));
        let mut ok_cts = BTreeMap::new();
        ok_cts.insert(
            "dataset://ns/a".to_string(),
            CONTENT_TYPE_OCTET_STREAM.to_string(),
        );
        assert!(build_physical_cas_bindings(
            &node_ok,
            &[AssetInstanceV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint_v0: "ab".repeat(32),
                uri: Some(valid_uri.clone()),
            }],
            &ok_cts
        )
        .is_ok());

        // Rejects non-cas prefix (`blobs/sha256/...`), uppercase hex, undeclared key, duplicate key, missing content-type, duplicate declared output
        assert!(build_physical_cas_bindings(
            &node_ok,
            &[AssetInstanceV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint_v0: "ab".repeat(32),
                uri: Some(format!("blobs/sha256/{}", "ab".repeat(32))),
            }],
            &ok_cts
        )
        .is_err());
        assert!(build_physical_cas_bindings(
            &node_ok,
            &[AssetInstanceV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint_v0: "AB".repeat(32),
                uri: Some(format!("cas://sha256/{}", "AB".repeat(32))),
            }],
            &ok_cts
        )
        .is_err());
        assert!(build_physical_cas_bindings(
            &node_ok,
            &[AssetInstanceV1 {
                asset_key: "dataset://ns/extra".to_string(),
                fingerprint_v0: "ab".repeat(32),
                uri: Some(valid_uri.clone()),
            }],
            &ok_cts
        )
        .is_err());
        assert!(build_physical_cas_bindings(
            &node_ok,
            &[AssetInstanceV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint_v0: "ab".repeat(32),
                uri: Some(valid_uri.clone()),
            }],
            &BTreeMap::new()
        )
        .is_err());
        let dup_node = make_node(
            "node/dup_out",
            "passthrough",
            &[],
            &["dataset://ns/a", "dataset://ns/a"],
            ExecutionTrust::Core,
        );
        assert!(build_physical_cas_bindings(
            &dup_node,
            &[AssetInstanceV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint_v0: "ab".repeat(32),
                uri: Some(valid_uri),
            }],
            &ok_cts
        )
        .is_err());
        node_ok.outputs.push(AssetRefV1 {
            asset_key: "dataset://ns/b".to_string(),
            fingerprint: None,
        });
        assert!(build_physical_cas_bindings(
            &node_ok,
            &[AssetInstanceV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint_v0: "ab".repeat(32),
                uri: Some(format!("cas://sha256/{}", "ab".repeat(32))),
            }],
            &ok_cts
        )
        .is_err());

        // Wasm cache policy/registry bypass checks + ManifestRefreshPolicy::Always schema checks
        let mut wasm = WASM_MAGIC_AND_VERSION_V1.to_vec();
        wasm.extend_from_slice(&[1, 7, 1, 0x60, 2, 0x7f, 0x7f, 1, 0x7f]);
        wasm.extend_from_slice(&[3, 2, 1, 0]);
        wasm.extend_from_slice(&[5, 4, 1, 0x01, 1, 1]);
        wasm.extend_from_slice(&[7, 13, 1, 9]);
        wasm.extend_from_slice(b"transform");
        wasm.extend_from_slice(&[0x00, 0x00]);
        wasm.extend_from_slice(&[10, 6, 1, 4, 0, 0x41, 0, 0x0b]);
        let hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let hex_hash = hex_lower(&hash);
        let mut registry = BTreeMap::new();
        registry.insert(hash, wasm);
        let mut policy = SandboxedExtensionPolicyV1::with_allowed_hash(hash);
        policy.allowed_exports.insert("transform".to_string());

        let base = temp_dir("st15_wasm_policy_cache_and_schema");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let run_id = RunId::from_bytes([0xE8; 16]);
        let profile_always = ArtifactWriteProfile {
            snapshot_profile: SnapshotProfile::Strict,
            manifest_policy: ManifestRefreshPolicy::Always,
        };
        let mut session = DataOpsSession::create(
            &base,
            run_id,
            profile_always,
            LockAcquisitionPolicy::FailFast,
        )
        .unwrap();
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        let payload = b"raw-wasm-data";
        let digest: [u8; 32] = Sha256::digest(payload).into();
        write_once_cas_blob(&run_dir, &digest, payload).unwrap();
        let ingest = make_node("ingest/wasm", "ingest", &[], &[], ExecutionTrust::Core);
        session
            .register_source(
                "dataset://ns/raw",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: format!("cas://sha256/{}", hex_lower(&digest)),
                    content_type: CONTENT_TYPE_OCTET_STREAM.to_string(),
                    auth_mode: swarm_torch_core::dataops::AuthModeMarker::None,
                    etag_or_version: None,
                },
                None,
                &ingest,
            )
            .unwrap();

        let mut wasm_node = make_node(
            "node/wasm",
            "wasm_transform",
            &["dataset://ns/raw"],
            &["dataset://ns/wout"],
            ExecutionTrust::SandboxedExtension,
        );
        wasm_node
            .params
            .insert("module_sha256".to_string(), CanonValue::Str(hex_hash));
        wasm_node.params.insert(
            "export".to_string(),
            CanonValue::Str("transform".to_string()),
        );
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("wasm-policy-cache".to_string()),
            nodes: vec![wasm_node],
            edges: vec![],
        };

        let executor = Arc::new(CountingWasmExecutor {
            calls: AtomicUsize::new(0),
        });
        let runner = NativeOpRunner::new().with_wasm_executor(executor.clone());
        let populate = |session: &mut DataOpsSession, pol: &SandboxedExtensionPolicyV1| {
            execute_graph_with_isolation(
                &graph,
                session,
                &runner,
                &PermissivePolicy,
                &RunnerIsolationProfile::WasmSandboxedV1(pol.clone()),
                Some(&registry),
                run_id,
                test_clock,
            )
            .unwrap()
        };

        // 1. Initial run executes once cleanly and caches.
        let r1 = populate(&mut session, &policy);
        assert_eq!(r1.executed_nodes, vec!["node/wasm"]);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        let r1_hit = populate(&mut session, &policy);
        assert_eq!(r1_hit.cached_nodes, vec!["node/wasm"]);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);

        // 2. Tightening store limits (max_fuel: 100_000 -> 500) MUST NOT hit cache; forces re-execution!
        let mut tighter_policy = policy.clone();
        tighter_policy.limits.max_fuel = 500;
        let r_tighter = populate(&mut session, &tighter_policy);
        assert!(r_tighter.cached_nodes.is_empty());
        assert_eq!(r_tighter.executed_nodes, vec!["node/wasm"]);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 2);

        // 3. Empty Some(&BTreeMap::new()) MUST deny + invalidate cached output instead of hitting cache!
        let r_empty_reg = execute_graph_with_isolation(
            &graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::WasmSandboxedV1(tighter_policy.clone()),
            Some(&BTreeMap::new()),
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r_empty_reg.failed_nodes, vec!["node/wasm"]);
        assert!(r_empty_reg.cached_nodes.is_empty());
        assert!(session.fingerprint("dataset://ns/wout").is_none());

        // 4. Revoked allowed_module_hashes MUST deny + invalidate!
        assert_eq!(
            populate(&mut session, &policy).executed_nodes,
            vec!["node/wasm"]
        );
        let mut revoked_hash_pol = policy.clone();
        revoked_hash_pol.allowed_module_hashes.clear();
        let r_revoked_hash = populate(&mut session, &revoked_hash_pol);
        assert_eq!(r_revoked_hash.failed_nodes, vec!["node/wasm"]);
        assert!(session.fingerprint("dataset://ns/wout").is_none());

        // 5. Revoked allowed_exports MUST deny + invalidate!
        assert_eq!(
            populate(&mut session, &policy).executed_nodes,
            vec!["node/wasm"]
        );
        let mut revoked_exp_pol = policy.clone();
        revoked_exp_pol.allowed_exports.clear();
        revoked_exp_pol
            .allowed_exports
            .insert("other_export".to_string());
        let r_revoked_exp = populate(&mut session, &revoked_exp_pol);
        assert_eq!(r_revoked_exp.failed_nodes, vec!["node/wasm"]);
        assert!(session.fingerprint("dataset://ns/wout").is_none());

        // 6. max_fuel == 0 MUST deny + invalidate!
        assert_eq!(
            populate(&mut session, &policy).executed_nodes,
            vec!["node/wasm"]
        );
        let mut zero_fuel_pol = policy.clone();
        zero_fuel_pol.limits.max_fuel = 0;
        let r_zero_fuel = populate(&mut session, &zero_fuel_pol);
        assert_eq!(r_zero_fuel.failed_nodes, vec!["node/wasm"]);
        assert!(session.fingerprint("dataset://ns/wout").is_none());

        // 7. ExternalSandboxedStub MUST deny & invalidate previously cached output AND leave manifest + v1 schemas valid!
        assert_eq!(
            populate(&mut session, &policy).executed_nodes,
            vec!["node/wasm"]
        );
        let r_stub_inv = execute_graph_with_isolation(
            &graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            &RunnerIsolationProfile::ExternalSandboxedStub {
                timeout_ms: 1000,
                max_memory_bytes: 1_048_576,
            },
            Some(&registry),
            run_id,
            test_clock,
        )
        .unwrap();
        assert_eq!(r_stub_inv.failed_nodes, vec!["node/wasm"]);
        assert!(session.fingerprint("dataset://ns/wout").is_none());

        // Also invalidate `dataset://ns/raw` so `registry_updates.ndjson` becomes 0 bytes, and validate manifest + all v1 schemas!
        session
            .invalidate_node_outputs(&make_node(
                "ingest/wasm",
                "ingest",
                &[],
                &["dataset://ns/raw"],
                ExecutionTrust::Core,
            ))
            .unwrap();
        assert_eq!(
            fs::read(run_dir.join("datasets/registry_updates.ndjson"))
                .unwrap()
                .len(),
            0
        );
        session.sink().bundle().validate_manifest().unwrap();
        let verified = session
            .sink()
            .bundle()
            .verify_and_load(&crate::artifacts::TrustPolicy::LegacyPermissive)
            .unwrap();
        validate_verified_bundle_schemas(&verified).unwrap();

        let _ = fs::remove_dir_all(&base);
    }
}
