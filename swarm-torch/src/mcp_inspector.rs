//! Zero-port read-only strict MCP inspector for signed SwarmTorch run bundles (`W12-05`).
//!
//! Operates strictly over line-delimited JSON-RPC 2.0 on `stdin`/`stdout` (zero TCP/HTTP listeners),
//! refuses [`crate::artifacts::TrustPolicy::LegacyPermissive`], verifies Ed25519 signatures and SHA-256 manifest entries
//! on every request, rejects all path traversal and non-read-only operations, and reads exclusively
//! from verified bundle contents (`VerifiedBundleV1`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use swarm_torch_core::dataops::{MaterializationStatusV0, TrustClass};
use swarm_torch_core::run_graph::{
    node_def_hash_v1, node_id_from_key, op_hash_v0, validate_graph_v1, ExecutionTrust, NodeV1,
};

use crate::artifacts::dp_ledger::{DpLedgerEntryV1, DP_LEDGER_REL_PATH};
use crate::artifacts::{hex_lower, RunArtifactBundle, TrustPolicy, VerifiedBundleV1};
use crate::report::{is_node_unsafe, load_report_from_verified, LoadWarning, Report};
use crate::scheduler::topological_sort_nodes;

/// Hard upper bound for a single incoming JSON-RPC 2.0 request line (1 MiB).
pub const MAX_JSONRPC_LINE_BYTES: usize = 1024 * 1024;

const JSONRPC_PARSE_ERROR: i64 = -32700;
const JSONRPC_INVALID_REQUEST: i64 = -32600;
const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
const JSONRPC_INVALID_PARAMS: i64 = -32602;
const JSONRPC_BUNDLE_VERIFICATION_FAILED: i64 = -32001;
const JSONRPC_LINE_TOO_LARGE: i64 = -32002;

/// Strict read-only MCP inspector backed by a single verified [`RunArtifactBundle`].
#[derive(Debug, Clone)]
pub struct McpInspectorServerV1 {
    run_dir: PathBuf,
    policy: TrustPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceKind {
    Manifest,
    Report,
    Graph,
    Lineage,
}

impl ResourceKind {
    fn from_suffix(s: &str) -> Option<Self> {
        match s {
            "manifest" => Some(Self::Manifest),
            "report" => Some(Self::Report),
            "graph" => Some(Self::Graph),
            "lineage" => Some(Self::Lineage),
            _ => None,
        }
    }
}

impl McpInspectorServerV1 {
    /// Construct a strict inspector server.
    ///
    /// Fails closed if `policy` is [`TrustPolicy::LegacyPermissive`] or has an empty keyset.
    pub fn try_new(run_dir: impl Into<PathBuf>, policy: TrustPolicy) -> io::Result<Self> {
        match &policy {
            TrustPolicy::Strict {
                trusted_public_keys,
                ..
            } if !trusted_public_keys.is_empty() => Ok(Self {
                run_dir: run_dir.into(),
                policy,
            }),
            TrustPolicy::Strict { .. } => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "strict MCP inspector requires at least one trusted Ed25519 public key",
            )),
            TrustPolicy::LegacyPermissive => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "MCP inspector refuses TrustPolicy::LegacyPermissive; TrustPolicy::Strict is mandatory",
            )),
        }
    }

    /// Construct a strict inspector server pinned to a single Ed25519 public key.
    pub fn strict_with_key(run_dir: impl Into<PathBuf>, public_key: [u8; 32]) -> io::Result<Self> {
        Self::try_new(run_dir, TrustPolicy::strict_with_key(public_key))
    }

    /// Path to the target run bundle directory.
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Serve newline-delimited JSON-RPC 2.0 requests strictly over `reader`/`writer` (`stdio`).
    ///
    /// Never opens any network socket or writes any artifact to disk.
    pub fn serve_stdio<R: BufRead, W: Write>(
        &self,
        mut reader: R,
        mut writer: W,
    ) -> io::Result<()> {
        loop {
            let line_opt = match read_bounded_line(&mut reader, MAX_JSONRPC_LINE_BYTES)? {
                BoundedLine::Eof => None,
                BoundedLine::Line(line) => Some(line),
                BoundedLine::TooLarge => {
                    let err_resp = jsonrpc_error(
                        Value::Null,
                        JSONRPC_LINE_TOO_LARGE,
                        format!("JSON-RPC request exceeds {MAX_JSONRPC_LINE_BYTES} byte limit"),
                    );
                    let encoded = serde_json::to_string(&err_resp).map_err(io::Error::other)?;
                    writer.write_all(encoded.as_bytes())?;
                    writer.write_all(b"\n")?;
                    writer.flush()?;
                    continue;
                }
            };
            let Some(line) = line_opt else {
                break;
            };
            if let Some(resp) = self.handle_jsonrpc_line(&line) {
                writer.write_all(resp.as_bytes())?;
                writer.write_all(b"\n")?;
                writer.flush()?;
            }
        }
        Ok(())
    }

    /// Process a single JSON-RPC 2.0 line; returns `None` for blank lines and notifications.
    pub fn handle_jsonrpc_line(&self, line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        if trimmed.len() > MAX_JSONRPC_LINE_BYTES {
            let resp = jsonrpc_error(
                Value::Null,
                JSONRPC_LINE_TOO_LARGE,
                format!("JSON-RPC request exceeds {MAX_JSONRPC_LINE_BYTES} byte limit"),
            );
            return serde_json::to_string(&resp).ok();
        }
        let parsed: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(err) => {
                let resp = jsonrpc_error(
                    Value::Null,
                    JSONRPC_PARSE_ERROR,
                    format!("malformed JSON-RPC payload: {err}"),
                );
                return serde_json::to_string(&resp).ok();
            }
        };
        self.handle_jsonrpc_value(&parsed)
            .and_then(|v| serde_json::to_string(&v).ok())
    }

    /// Process a parsed JSON-RPC 2.0 request object; returns `None` for notifications.
    pub fn handle_jsonrpc_value(&self, req: &Value) -> Option<Value> {
        let Some(obj) = req.as_object() else {
            return Some(jsonrpc_error(
                Value::Null,
                JSONRPC_INVALID_REQUEST,
                "JSON-RPC request must be a JSON object",
            ));
        };

        let has_id = obj.contains_key("id");
        let id = obj.get("id").cloned().unwrap_or(Value::Null);

        if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return has_id.then(|| {
                jsonrpc_error(
                    id,
                    JSONRPC_INVALID_REQUEST,
                    "unsupported or missing jsonrpc version (expected \"2.0\")",
                )
            });
        }
        let Some(method) = obj.get("method").and_then(Value::as_str) else {
            return has_id
                .then(|| jsonrpc_error(id, JSONRPC_INVALID_REQUEST, "missing string method"));
        };

        // Notifications without `id` return no response per JSON-RPC 2.0.
        if !has_id {
            return None;
        }

        let params = obj.get("params").cloned().unwrap_or_else(|| json!({}));
        let result = match method {
            "initialize" => Ok(handle_initialize()),
            "ping" => Ok(json!({})),
            "resources/list" => self.handle_resources_list(),
            "resources/read" => self.handle_resources_read(&params),
            "tools/list" => Ok(handle_tools_list()),
            "tools/call" => self.handle_tools_call(&params),
            _ => Err((
                JSONRPC_METHOD_NOT_FOUND,
                format!("method '{method}' is not exposed by read-only strict MCP inspector"),
            )),
        };

        Some(match result {
            Ok(res) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": res,
            }),
            Err((code, msg)) => jsonrpc_error(id, code, msg),
        })
    }

    fn verify_and_load(
        &self,
    ) -> Result<(VerifiedBundleV1, Report, Vec<LoadWarning>), (i64, String)> {
        if !matches!(self.policy, TrustPolicy::Strict { .. }) {
            return Err((
                JSONRPC_BUNDLE_VERIFICATION_FAILED,
                "strict policy invariant violated".to_string(),
            ));
        }
        let bundle = RunArtifactBundle::open(&self.run_dir).map_err(|e| {
            (
                JSONRPC_BUNDLE_VERIFICATION_FAILED,
                format!("failed to open run bundle: {e}"),
            )
        })?;
        let verified = bundle.verify_and_load(&self.policy).map_err(|e| {
            (
                JSONRPC_BUNDLE_VERIFICATION_FAILED,
                format!("strict signature/digest verification failed: {e}"),
            )
        })?;
        if !verified.is_strict_verified() {
            return Err((
                JSONRPC_BUNDLE_VERIFICATION_FAILED,
                "bundle did not achieve VerifiedStrict verification state".to_string(),
            ));
        }
        let (report, warnings) = load_report_from_verified(&verified).map_err(|e| {
            (
                JSONRPC_BUNDLE_VERIFICATION_FAILED,
                format!("failed to decode verified run report: {e}"),
            )
        })?;
        Ok((verified, report, warnings))
    }

    fn handle_resources_list(&self) -> Result<Value, (i64, String)> {
        let (verified, _, _) = self.verify_and_load()?;
        let run_id_hex = hex_lower(verified.run_id().as_bytes());
        let resources = vec![
            resource_entry(
                &format!("swarmtorch://runs/{run_id_hex}/manifest"),
                "manifest.json",
                "Cryptographically verified SwarmTorch bundle manifest",
            ),
            resource_entry(
                &format!("swarmtorch://runs/{run_id_hex}/report"),
                "report.json",
                "Aggregated verified run report (graph, materializations, spans, events, metrics)",
            ),
            resource_entry(
                &format!("swarmtorch://runs/{run_id_hex}/graph"),
                "graph.json",
                "Canonical normalized execution DAG (GraphV1)",
            ),
            resource_entry(
                &format!("swarmtorch://runs/{run_id_hex}/lineage"),
                "lineage.json",
                "Verified dataset lineage graph (DatasetLineageV1)",
            ),
        ];
        Ok(json!({ "resources": resources }))
    }

    fn handle_resources_read(&self, params: &Value) -> Result<Value, (i64, String)> {
        let uri = params.get("uri").and_then(Value::as_str).ok_or((
            JSONRPC_INVALID_PARAMS,
            "resources/read missing params.uri string".to_string(),
        ))?;

        let (verified, report, _warnings) = self.verify_and_load()?;
        let expected_run_id = hex_lower(verified.run_id().as_bytes());
        let kind = parse_strict_resource_uri(uri, &expected_run_id)?;

        let payload: Value = match kind {
            ResourceKind::Manifest => serde_json::to_value(verified.manifest())
                .map_err(|e| (JSONRPC_BUNDLE_VERIFICATION_FAILED, e.to_string()))?,
            ResourceKind::Report => serde_json::to_value(&report)
                .map_err(|e| (JSONRPC_BUNDLE_VERIFICATION_FAILED, e.to_string()))?,
            ResourceKind::Graph => {
                let raw = verified
                    .require_bytes("graph.json")
                    .map_err(|e| (JSONRPC_BUNDLE_VERIFICATION_FAILED, e.to_string()))?;
                serde_json::from_slice(raw)
                    .map_err(|e| (JSONRPC_BUNDLE_VERIFICATION_FAILED, e.to_string()))?
            }
            ResourceKind::Lineage => serde_json::to_value(&report.lineage)
                .map_err(|e| (JSONRPC_BUNDLE_VERIFICATION_FAILED, e.to_string()))?,
        };

        let text = serde_json::to_string_pretty(&payload)
            .map_err(|e| (JSONRPC_BUNDLE_VERIFICATION_FAILED, e.to_string()))?;

        Ok(json!({
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": text,
            }]
        }))
    }

    fn handle_tools_call(&self, params: &Value) -> Result<Value, (i64, String)> {
        let name = params.get("name").and_then(Value::as_str).ok_or((
            JSONRPC_INVALID_PARAMS,
            "tools/call missing params.name string".to_string(),
        ))?;
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let (verified, report, warnings) = self.verify_and_load()?;

        let structured = match name {
            "inspect_run_report" => tool_inspect_run_report(&verified, &report, &warnings),
            "audit_graph_trust_and_cycles" => tool_audit_graph_trust_and_cycles(&verified, &report),
            "predict_node_outputs" => tool_predict_node_outputs(&report, &arguments)?,
            "trace_aggregation_budget" => tool_trace_aggregation_budget(&verified, &report)?,
            _ => {
                return Err((
                    JSONRPC_INVALID_PARAMS,
                    format!("unknown or non-read-only tool '{name}' rejected"),
                ));
            }
        };

        let text = serde_json::to_string_pretty(&structured)
            .map_err(|e| (JSONRPC_BUNDLE_VERIFICATION_FAILED, e.to_string()))?;

        Ok(json!({
            "content": [{
                "type": "text",
                "text": text,
            }],
            "structuredContent": structured,
            "isError": false,
        }))
    }
}

fn handle_initialize() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "resources": {
                "subscribe": false,
                "listChanged": false,
            },
            "tools": {
                "listChanged": false,
            },
        },
        "serverInfo": {
            "name": "swarm-torch-mcp-inspector",
            "version": env!("CARGO_PKG_VERSION"),
            "transport": "stdio",
            "readOnly": true,
        },
    })
}

fn handle_tools_list() -> Value {
    let ro_annotations = json!({
        "readOnlyHint": true,
        "destructiveHint": false,
        "idempotentHint": true,
        "openWorldHint": false,
    });
    json!({
        "tools": [
            {
                "name": "inspect_run_report",
                "description": "Inspect strictly verified manifest, nodes, materializations, spans, events, and metrics for the run bundle.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                },
                "annotations": ro_annotations,
            },
            {
                "name": "audit_graph_trust_and_cycles",
                "description": "Audit normalized GraphV1 for validation errors, dependency cycles, unsafe surfaces, trust classes, and strict Ed25519 manifest verification.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                },
                "annotations": ro_annotations,
            },
            {
                "name": "predict_node_outputs",
                "description": "Predict output asset keys, deterministic node hashes, and input readiness for a node (or all nodes) in the verified run graph.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "node_key": {
                            "type": "string",
                            "description": "Optional specific node_key to inspect"
                        }
                    }
                },
                "annotations": ro_annotations,
            },
            {
                "name": "trace_aggregation_budget",
                "description": "Trace verified RDP privacy ledger entries (dp/ledger.ndjson) and DP/aggregation metrics recorded inside the signed bundle.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {}
                },
                "annotations": ro_annotations,
            },
        ]
    })
}

fn resource_entry(uri: &str, name: &str, description: &str) -> Value {
    json!({
        "uri": uri,
        "name": name,
        "description": description,
        "mimeType": "application/json",
    })
}

fn parse_strict_resource_uri(
    uri: &str,
    expected_run_id: &str,
) -> Result<ResourceKind, (i64, String)> {
    const PREFIX: &str = "swarmtorch://runs/";
    if uri.contains("..")
        || uri.contains('\\')
        || uri.contains('\0')
        || uri.contains('?')
        || uri.contains('#')
        || uri.chars().any(char::is_whitespace)
    {
        return Err((
            JSONRPC_INVALID_PARAMS,
            "resource URI rejected by path traversal and character guard".to_string(),
        ));
    }
    let Some(rest) = uri.strip_prefix(PREFIX) else {
        return Err((
            JSONRPC_INVALID_PARAMS,
            format!("resource URI must start with '{PREFIX}'"),
        ));
    };
    let mut parts = rest.split('/');
    let Some(run_id_part) = parts.next() else {
        return Err((
            JSONRPC_INVALID_PARAMS,
            "missing run_id in resource URI".to_string(),
        ));
    };
    let Some(kind_part) = parts.next() else {
        return Err((
            JSONRPC_INVALID_PARAMS,
            "missing resource kind in resource URI".to_string(),
        ));
    };
    if parts.next().is_some() {
        return Err((
            JSONRPC_INVALID_PARAMS,
            "unexpected extra path segments in resource URI".to_string(),
        ));
    }
    if run_id_part != expected_run_id {
        return Err((
            JSONRPC_INVALID_PARAMS,
            format!(
                "run_id '{run_id_part}' does not match verified bundle run_id '{expected_run_id}'"
            ),
        ));
    }
    ResourceKind::from_suffix(kind_part).ok_or_else(|| {
        (
            JSONRPC_INVALID_PARAMS,
            format!("unsupported read-only resource '{kind_part}'"),
        )
    })
}

fn tool_inspect_run_report(
    verified: &VerifiedBundleV1,
    report: &Report,
    warnings: &[LoadWarning],
) -> Value {
    let run_id_hex = hex_lower(verified.run_id().as_bytes());
    let mut failed_materializations = 0usize;
    let mut skipped_materializations = 0usize;
    for m in &report.materializations {
        match m.status {
            MaterializationStatusV0::Ok => {}
            MaterializationStatusV0::Error => failed_materializations += 1,
            MaterializationStatusV0::Skipped => skipped_materializations += 1,
        }
    }
    let node_keys: Vec<&str> = report
        .graph
        .nodes
        .iter()
        .map(|n| n.node_key.as_str())
        .collect();
    let warning_messages: Vec<String> = warnings.iter().map(|w| format!("{w:?}")).collect();

    json!({
        "run_id": run_id_hex,
        "verified_strict": verified.is_strict_verified(),
        "manifest_entries": verified.manifest().entries.len(),
        "node_count": report.graph.nodes.len(),
        "node_keys": node_keys,
        "registered_datasets": report.registry.datasets.len(),
        "lineage_edges": report.lineage.edges.len(),
        "materialization_count": report.materializations.len(),
        "failed_materializations": failed_materializations,
        "skipped_materializations": skipped_materializations,
        "span_count": report.spans.len(),
        "event_count": report.events.len(),
        "metric_count": report.metrics.len(),
        "warnings": warning_messages,
    })
}

fn tool_audit_graph_trust_and_cycles(verified: &VerifiedBundleV1, report: &Report) -> Value {
    let validation_error = validate_graph_v1(&report.graph)
        .err()
        .map(|e| e.to_string());
    let (cycle_free, topological_order, topo_error) = match topological_sort_nodes(&report.graph) {
        Ok(ordered) => (
            true,
            ordered.into_iter().map(|n| n.node_key).collect::<Vec<_>>(),
            None,
        ),
        Err(err) => (false, Vec::new(), Some(err.to_string())),
    };

    let mut unsafe_nodes = Vec::new();
    let mut sandboxed_extension_nodes = Vec::new();
    let mut unsafe_extension_nodes = Vec::new();
    for node in &report.graph.nodes {
        if is_node_unsafe(node, &report.registry) {
            unsafe_nodes.push(node.node_key.as_str());
        }
        match node.execution_trust {
            ExecutionTrust::Core => {}
            ExecutionTrust::SandboxedExtension => {
                sandboxed_extension_nodes.push(node.node_key.as_str());
            }
            ExecutionTrust::UnsafeExtension => {
                unsafe_extension_nodes.push(node.node_key.as_str());
            }
        }
    }

    let untrusted_datasets: Vec<&str> = report
        .registry
        .datasets
        .iter()
        .filter(|d| d.trust == TrustClass::Untrusted)
        .map(|d| d.asset_key.as_str())
        .collect();

    json!({
        "strict_signature_verified": verified.is_strict_verified(),
        "graph_valid": validation_error.is_none(),
        "validation_error": validation_error,
        "cycle_free": cycle_free,
        "topological_error": topo_error,
        "topological_order": topological_order,
        "unsafe_nodes": unsafe_nodes,
        "sandboxed_extension_nodes": sandboxed_extension_nodes,
        "unsafe_extension_nodes": unsafe_extension_nodes,
        "untrusted_datasets": untrusted_datasets,
    })
}

fn tool_predict_node_outputs(report: &Report, arguments: &Value) -> Result<Value, (i64, String)> {
    let target_key = match arguments.get("node_key") {
        Some(Value::String(s)) => Some(s.as_str()),
        Some(Value::Null) | None => None,
        Some(_) => {
            return Err((
                JSONRPC_INVALID_PARAMS,
                "predict_node_outputs 'node_key' argument must be a string".to_string(),
            ));
        }
    };

    let known_assets: BTreeSet<&str> = report
        .registry
        .datasets
        .iter()
        .map(|d| d.asset_key.as_str())
        .chain(report.materializations.iter().map(|m| m.asset_key.as_str()))
        .collect();

    let mut latest_mat_by_asset: BTreeMap<&str, _> = BTreeMap::new();
    for m in &report.materializations {
        latest_mat_by_asset.insert(m.asset_key.as_str(), m);
    }

    let selected: Vec<&NodeV1> = match target_key {
        Some(key) => {
            let found = report
                .graph
                .nodes
                .iter()
                .find(|n| n.node_key == key)
                .ok_or_else(|| {
                    (
                        JSONRPC_INVALID_PARAMS,
                        format!("node_key '{key}' not found in verified graph"),
                    )
                })?;
            vec![found]
        }
        None => report.graph.nodes.iter().collect(),
    };

    let mut predictions = Vec::with_capacity(selected.len());
    for node in selected {
        let missing_inputs: Vec<&str> = node
            .inputs
            .iter()
            .map(|i| i.asset_key.as_str())
            .filter(|k| !known_assets.contains(k))
            .collect();
        let computed_node_def_hash = node_def_hash_v1(node).ok().map(|h| hex_lower(&h));
        let computed_op_hash = op_hash_v0(node).ok().map(|h| hex_lower(&h));
        let expected_outputs: Vec<Value> = node
            .outputs
            .iter()
            .map(|o| {
                let prior = latest_mat_by_asset.get(o.asset_key.as_str());
                json!({
                    "asset_key": o.asset_key,
                    "observed_rows": prior.and_then(|m| m.rows),
                    "observed_bytes": prior.and_then(|m| m.bytes),
                    "last_status": prior.map(|m| &m.status),
                })
            })
            .collect();
        let resolved_node_id = node
            .node_id
            .unwrap_or_else(|| node_id_from_key(&node.node_key));
        predictions.push(json!({
            "node_key": node.node_key,
            "node_id": resolved_node_id.to_string(),
            "op_type": node.op_type,
            "execution_trust": node.execution_trust,
            "unsafe_surface": is_node_unsafe(node, &report.registry),
            "all_inputs_available": missing_inputs.is_empty(),
            "missing_inputs": missing_inputs,
            "computed_node_def_hash": computed_node_def_hash,
            "computed_op_hash": computed_op_hash,
            "predicted_outputs": expected_outputs,
        }));
    }

    Ok(json!({
        "run_graph_id": report.graph.graph_id,
        "node_count": predictions.len(),
        "predictions": predictions,
    }))
}

fn tool_trace_aggregation_budget(
    verified: &VerifiedBundleV1,
    report: &Report,
) -> Result<Value, (i64, String)> {
    // Only trust ledger bytes verified inside the Ed25519-signed manifest!
    let ledger_entries: Vec<DpLedgerEntryV1> = match verified.get_bytes(DP_LEDGER_REL_PATH) {
        Some(raw) => {
            let text = std::str::from_utf8(raw).map_err(|e| {
                (
                    JSONRPC_BUNDLE_VERIFICATION_FAILED,
                    format!("verified dp ledger is invalid UTF-8: {e}"),
                )
            })?;
            let mut entries = Vec::new();
            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let entry: DpLedgerEntryV1 = serde_json::from_str(trimmed).map_err(|e| {
                    (
                        JSONRPC_BUNDLE_VERIFICATION_FAILED,
                        format!("verified dp ledger contains corrupt NDJSON record: {e}"),
                    )
                })?;
                entries.push(entry);
            }
            entries
        }
        None => Vec::new(),
    };

    let latest_ledger = ledger_entries.last();
    let dp_metrics: Vec<Value> = report
        .metrics
        .iter()
        .filter(|m| {
            m.name.starts_with("dp.")
                || m.name.starts_with("rdp.")
                || m.name.starts_with("coordinator.")
                || m.name.contains("epsilon")
                || m.name.contains("round")
        })
        .map(|m| {
            json!({
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "ts_unix_nanos": m.ts_unix_nanos,
            })
        })
        .collect();

    Ok(json!({
        "strict_signature_verified": verified.is_strict_verified(),
        "dp_ledger_present_in_signed_manifest": verified.get_bytes(DP_LEDGER_REL_PATH).is_some(),
        "dp_ledger_entry_count": ledger_entries.len(),
        "committed_rounds": latest_ledger.map(|e| e.committed_rounds).unwrap_or(0),
        "latest_converted_epsilon": latest_ledger.map(|e| e.converted_epsilon),
        "latest_optimal_alpha": latest_ledger.map(|e| e.optimal_alpha),
        "target_delta": latest_ledger.map(|e| e.target_delta),
        "cumulative_rdp_orders": latest_ledger.map(|e| &e.cumulative_rdp_orders),
        "dp_ledger_entries": ledger_entries,
        "aggregation_metrics": dp_metrics,
    }))
}

fn jsonrpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        },
    })
}

enum BoundedLine {
    Eof,
    Line(String),
    TooLarge,
}

fn read_bounded_line<R: BufRead>(reader: &mut R, max_bytes: usize) -> io::Result<BoundedLine> {
    let mut buf = Vec::new();
    let mut oversized = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            if buf.is_empty() && !oversized {
                return Ok(BoundedLine::Eof);
            }
            if oversized {
                return Ok(BoundedLine::TooLarge);
            }
            let text = String::from_utf8(buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            return Ok(BoundedLine::Line(text));
        }

        if let Some(nl_pos) = available.iter().position(|&b| b == b'\n') {
            if !oversized && buf.len().saturating_add(nl_pos) <= max_bytes {
                buf.extend_from_slice(&available[..nl_pos]);
            } else {
                oversized = true;
            }
            reader.consume(nl_pos + 1);
            if oversized {
                return Ok(BoundedLine::TooLarge);
            }
            if buf.ends_with(b"\r") {
                buf.pop();
            }
            let text = String::from_utf8(buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            return Ok(BoundedLine::Line(text));
        }

        let len = available.len();
        if !oversized && buf.len().saturating_add(len) <= max_bytes {
            buf.extend_from_slice(available);
        } else {
            oversized = true;
        }
        reader.consume(len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::dp_ledger::pre_release_debit_dp_accountant;
    use std::fs;
    use swarm_torch_core::crypto::KeyPair;
    use swarm_torch_core::dataops::{
        CacheDecisionV0, DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1, LineageEdgeV1,
        MaterializationRecordV2, TransformAuditV0, UnsafeReasonV0,
    };
    use swarm_torch_core::dp::{
        DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1, RdpAccountantV1,
    };
    use swarm_torch_core::observe::{AttrMap, MetricRecord, RunId};
    use swarm_torch_core::run_graph::{
        AssetRefV1, CanonParams, ExecutionTrust, GraphV1, NodeV1, OpKind,
    };

    fn temp_bundle_dir(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "swarmtorch_w12_mcp_{}_{}",
            label,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn build_signed_test_bundle(label: &str) -> (PathBuf, PathBuf, KeyPair, RunId, GraphV1) {
        let base = temp_bundle_dir(label);
        let run_id = RunId::from_bytes([0x5a; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();

        let ingest = NodeV1 {
            node_key: "prep/ingest".to_string(),
            node_id: Some(node_id_from_key("prep/ingest")),
            op_kind: OpKind::Data,
            op_type: "ingest".to_string(),
            inputs: vec![],
            outputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/raw".to_string(),
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
        };
        let wasm_node = NodeV1 {
            node_key: "ext/normalize".to_string(),
            node_id: Some(node_id_from_key("ext/normalize")),
            op_kind: OpKind::Data,
            op_type: "wasm_transform".to_string(),
            inputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/raw".to_string(),
                fingerprint: None,
            }],
            outputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/norm".to_string(),
                fingerprint: None,
            }],
            params: CanonParams::new(),
            code_ref: Some("wasm://norm".to_string()),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::SandboxedExtension,
            node_def_hash: None,
            execution_hint: None,
            cache_policy: None,
            materialization_policy: None,
            resources: None,
            op_hash: None,
        };
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("w12-mcp-test-graph".to_string()),
            nodes: vec![ingest, wasm_node],
            edges: vec![],
        }
        .normalize()
        .unwrap();
        bundle.write_graph(&graph).unwrap();

        let registry = DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![DatasetEntryV1 {
                asset_key: "dataset://ns/raw".to_string(),
                fingerprint_v0: "aa".repeat(32),
                source_fingerprint_v0: "bb".repeat(32),
                schema_hash_v0: "cc".repeat(32),
                recipe_hash_v0: "dd".repeat(32),
                trust: TrustClass::Untrusted,
                source: None,
                schema: None,
                license_flags: vec![],
                pii_tags: vec![],
            }],
        };
        bundle.write_dataset_registry(&registry).unwrap();

        let lineage = DatasetLineageV1 {
            schema_version: 1,
            edges: vec![LineageEdgeV1 {
                input_fingerprint_v0: "aa".repeat(32),
                output_fingerprint_v0: "ee".repeat(32),
                node_id: node_id_from_key("ext/normalize"),
                op_kind: OpKind::Data,
            }],
        };
        bundle.write_dataset_lineage(&lineage).unwrap();

        bundle
            .append_materialization_v2(&MaterializationRecordV2 {
                schema_version: 2,
                record_seq: 1,
                ts_unix_nanos: 1_000,
                asset_key: "dataset://ns/norm".to_string(),
                fingerprint_v0: "ee".repeat(32),
                node_id: node_id_from_key("ext/normalize"),
                node_def_hash: "ff".repeat(32),
                op_type: "wasm_transform".to_string(),
                input_asset_keys: vec!["dataset://ns/raw".to_string()],
                input_fingerprints_v0: vec!["aa".repeat(32)],
                rows: Some(42),
                bytes: Some(1024),
                duration_ms: Some(3),
                cache_decision: CacheDecisionV0::Miss,
                cache_reason: Some("first_run".to_string()),
                cache_key_v0: Some("01".repeat(32)),
                cache_hit: Some(false),
                unsafe_surface: true,
                unsafe_reasons: vec![UnsafeReasonV0::UntrustedInput],
                applied_transforms: vec![TransformAuditV0 {
                    transform_name: "clip_normalize".to_string(),
                    core_trusted: true,
                    round_id: 1,
                }],
                status: MaterializationStatusV0::Ok,
                error_code: None,
                quality: None,
            })
            .unwrap();

        bundle
            .append_metric(&MetricRecord {
                schema_version: 1,
                ts_unix_nanos: 2_000,
                trace_id: swarm_torch_core::observe::TraceId::from_bytes([0x5a; 16]),
                span_id: None,
                name: "dp.round_epsilon".to_string(),
                value: 0.45,
                unit: Some("epsilon".to_string()),
                attrs: AttrMap::new(),
            })
            .unwrap();

        // Append durable pre-release DP ledger entry BEFORE signing manifest.
        let mut accountant = RdpAccountantV1::new(DpMechanismProfileV1 {
            unit: DpUnitV1::ClientLevel,
            adjacency: DpAdjacencyV1::AddRemoveOne,
            clipping_radius_tau: 1.0,
            noise_multiplier: 1.2,
            target_delta: 1e-5,
            max_epsilon: 3.0,
            conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        })
        .unwrap();
        pre_release_debit_dp_accountant(
            bundle.run_dir(),
            run_id,
            1,
            &"00".repeat(32),
            &mut accountant,
            1.0,
            0.1,
            2_500,
        )
        .unwrap();

        let signer = KeyPair::from_seed([0x7b; 32]).unwrap();
        bundle.finalize_signed_manifest(&signer).unwrap();
        let run_dir = bundle.run_dir().to_path_buf();
        (base, run_dir, signer, run_id, graph)
    }

    #[test]
    fn mcp_inspector_refuses_legacy_permissive_and_tampered_bundles() {
        let (base, run_dir, signer, _run_id, _graph) = build_signed_test_bundle("fail_closed");
        assert!(McpInspectorServerV1::try_new(&run_dir, TrustPolicy::LegacyPermissive).is_err());

        let server = McpInspectorServerV1::strict_with_key(&run_dir, *signer.public_key()).unwrap();

        // Wrong public key fails closed on resource read and tool call.
        let wrong_key = KeyPair::from_seed([0x99; 32]).unwrap();
        let bad_server =
            McpInspectorServerV1::strict_with_key(&run_dir, *wrong_key.public_key()).unwrap();
        let bad_resp: Value = serde_json::from_str(
            &bad_server
                .handle_jsonrpc_line(
                    r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"inspect_run_report","arguments":{}}}"#,
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            bad_resp["error"]["code"],
            JSONRPC_BUNDLE_VERIFICATION_FAILED
        );

        // Tamper a manifest-covered file on disk -> verified bundle check fails closed!
        fs::write(run_dir.join("metrics.ndjson"), b"{\"tampered\":true}\n").unwrap();
        let tampered_resp: Value = serde_json::from_str(
            &server
                .handle_jsonrpc_line(r#"{"jsonrpc":"2.0","id":2,"method":"resources/list"}"#)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            tampered_resp["error"]["code"],
            JSONRPC_BUNDLE_VERIFICATION_FAILED
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn mcp_inspector_exposes_all_resources_tools_and_blocks_mutations_and_traversal() {
        let (base, run_dir, signer, run_id, _graph) = build_signed_test_bundle("full_flow");
        let server = McpInspectorServerV1::strict_with_key(&run_dir, *signer.public_key()).unwrap();
        let run_id_hex = hex_lower(run_id.as_bytes());

        // Initialize advertises strict readOnly stdio transport
        let init_resp: Value = serde_json::from_str(
            &server
                .handle_jsonrpc_line(
                    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(init_resp["result"]["serverInfo"]["readOnly"], true);
        assert_eq!(
            init_resp["result"]["capabilities"]["resources"]["subscribe"],
            false
        );

        // Notifications omit id and yield None
        assert!(server
            .handle_jsonrpc_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .is_none());

        // List all 4 read-only resources
        let list_res: Value = serde_json::from_str(
            &server
                .handle_jsonrpc_line(r#"{"jsonrpc":"2.0","id":2,"method":"resources/list"}"#)
                .unwrap(),
        )
        .unwrap();
        let resources = list_res["result"]["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 4);

        // Read all 4 resources
        for suffix in ["manifest", "report", "graph", "lineage"] {
            let uri = format!("swarmtorch://runs/{run_id_hex}/{suffix}");
            let req = json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/read",
                "params": { "uri": uri },
            })
            .to_string();
            let resp: Value =
                serde_json::from_str(&server.handle_jsonrpc_line(&req).unwrap()).unwrap();
            assert!(
                resp["error"].is_null(),
                "unexpected error for {suffix}: {resp}"
            );
            let text = resp["result"]["contents"][0]["text"].as_str().unwrap();
            assert!(!text.is_empty());
        }

        // Reject traversal & cross-run URI
        for bad_uri in [
            format!("swarmtorch://runs/{run_id_hex}/../../etc/passwd"),
            format!("swarmtorch://runs/{}/manifest", "ff".repeat(16)),
            "file:///etc/passwd".to_string(),
        ] {
            let req = json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": { "uri": bad_uri },
            })
            .to_string();
            let resp: Value =
                serde_json::from_str(&server.handle_jsonrpc_line(&req).unwrap()).unwrap();
            assert_eq!(resp["error"]["code"], JSONRPC_INVALID_PARAMS);
        }

        // Reject mutation / unknown methods & unknown tools
        for bad_method in ["resources/write", "tools/create", "fs/write", "shutdown"] {
            let req = json!({ "jsonrpc": "2.0", "id": 5, "method": bad_method }).to_string();
            let resp: Value =
                serde_json::from_str(&server.handle_jsonrpc_line(&req).unwrap()).unwrap();
            assert_eq!(resp["error"]["code"], JSONRPC_METHOD_NOT_FOUND);
        }
        let bad_tool: Value = serde_json::from_str(
            &server
                .handle_jsonrpc_line(
                    r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"mutate_bundle","arguments":{}}}"#,
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(bad_tool["error"]["code"], JSONRPC_INVALID_PARAMS);

        // Call all 4 registered read-only tools via serve_stdio loop
        let reqs = [
            json!({"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"inspect_run_report","arguments":{}}}).to_string(),
            json!({"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"audit_graph_trust_and_cycles","arguments":{}}}).to_string(),
            json!({"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"predict_node_outputs","arguments":{"node_key":"ext/normalize"}}}).to_string(),
            json!({"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"trace_aggregation_budget","arguments":{}}}).to_string(),
        ];
        let input_stream = reqs.join("\n") + "\n";
        let mut output_buf = Vec::new();
        server
            .serve_stdio(io::Cursor::new(input_stream.as_bytes()), &mut output_buf)
            .unwrap();

        let lines: Vec<Value> = String::from_utf8(output_buf)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0]["result"]["structuredContent"]["node_count"], 2);
        assert_eq!(lines[1]["result"]["structuredContent"]["cycle_free"], true);
        assert_eq!(
            lines[1]["result"]["structuredContent"]["sandboxed_extension_nodes"],
            json!(["ext/normalize"])
        );
        assert_eq!(
            lines[1]["result"]["structuredContent"]["untrusted_datasets"],
            json!(["dataset://ns/raw"])
        );
        assert_eq!(
            lines[2]["result"]["structuredContent"]["predictions"][0]["predicted_outputs"][0]
                ["observed_rows"],
            42
        );
        assert_eq!(
            lines[3]["result"]["structuredContent"]["dp_ledger_present_in_signed_manifest"],
            true
        );
        assert_eq!(
            lines[3]["result"]["structuredContent"]["committed_rounds"],
            1
        );

        let _ = fs::remove_dir_all(&base);
    }
}
