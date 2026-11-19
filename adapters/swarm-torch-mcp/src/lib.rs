#![forbid(unsafe_code)]

//! Isolated zero-port read-only strict MCP inspector adapter wrapping
//! [`swarm_torch::mcp_inspector::McpInspectorServerV1`] on the official `rmcp 3.11` SDK
//! (`ADR-0016`, `15D-5`).
//!
//! **Maturity Tier:** `Integration verified` — the official `rmcp 3.11` SDK serves the MCP
//! protocol (capability negotiation, framing, lifecycle, cancellation) over stdio through
//! [`SwarmTorchMcpAdapterV1`]'s [`ServerHandler`] implementation, which delegates every tool and
//! resource call to the canonical read-only [`McpInspectorServerV1`] so trust policy, schema
//! validation, and bounded DP ledger tracing (`AR-14-04`) stay in one place.
//!
//! - Incoming lines are capped at [`MAX_JSONRPC_LINE_BYTES`] (1 MiB) on both paths:
//!   [`SwarmTorchMcpAdapterV1::serve_rmcp`] / [`SwarmTorchMcpAdapterV1::serve_rmcp_stdio`] configure
//!   `rmcp`'s [`AsyncRwTransport::with_max_line_length`] (dropping oversized lines without buffering
//!   while keeping the session open), whereas the legacy synchronous
//!   [`SwarmTorchMcpAdapterV1::serve_stdio`] emits JSON-RPC error `-32002`.
//! - Concurrent blocking verification tasks spawned by `rmcp`'s per-request dispatcher are bounded
//!   by a semaphore ([`DEFAULT_MAX_IN_FLIGHT_REQUESTS`]) and honor `RequestContext` cancellation.
//! - Differential privacy tool fields surface experimental accounting-only metadata (`ST-02`).

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ErrorCode, ErrorData, Implementation,
    ListResourceTemplatesResult, ListResourcesResult, ListToolsResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult, ResultType,
    ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::{QuitReason, RequestContext};
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::{RoleServer, ServerHandler, ServiceExt};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::Semaphore;

pub use swarm_torch::artifacts::TrustPolicy;
pub use swarm_torch::mcp_inspector::{McpInspectorServerV1, MAX_JSONRPC_LINE_BYTES};

/// Default upper bound on concurrent blocking bundle-verification tasks per adapter instance.
pub const DEFAULT_MAX_IN_FLIGHT_REQUESTS: usize = 8;

/// Adapter handle serving strict read-only SwarmTorch bundle inspection over MCP.
#[derive(Debug, Clone)]
pub struct SwarmTorchMcpAdapterV1 {
    inner: McpInspectorServerV1,
    in_flight: Arc<Semaphore>,
    max_in_flight: usize,
}

impl SwarmTorchMcpAdapterV1 {
    /// Construct an adapter bound to a run bundle directory and an explicit [`TrustPolicy`].
    ///
    /// Fails closed if `policy` is [`TrustPolicy::LegacyPermissive`] or has an empty keyset.
    pub fn try_new(run_dir: impl Into<PathBuf>, policy: TrustPolicy) -> io::Result<Self> {
        let inner = McpInspectorServerV1::try_new(run_dir, policy)?;
        Ok(Self {
            inner,
            in_flight: Arc::new(Semaphore::new(DEFAULT_MAX_IN_FLIGHT_REQUESTS)),
            max_in_flight: DEFAULT_MAX_IN_FLIGHT_REQUESTS,
        })
    }

    /// Construct an adapter bound to a verified run bundle directory and a strict Ed25519 key.
    pub fn strict_with_key(run_dir: impl Into<PathBuf>, public_key: [u8; 32]) -> io::Result<Self> {
        Self::try_new(run_dir, TrustPolicy::strict_with_key(public_key))
    }

    /// Configure the maximum number of concurrent blocking inspector requests
    /// (clamped to `1..=Semaphore::MAX_PERMITS`).
    pub fn with_max_in_flight_requests(mut self, max_in_flight: usize) -> Self {
        let bounded = max_in_flight.clamp(1, Semaphore::MAX_PERMITS);
        self.in_flight = Arc::new(Semaphore::new(bounded));
        self.max_in_flight = bounded;
        self
    }

    /// Return the configured maximum number of concurrent blocking inspector requests.
    pub const fn max_in_flight_requests(&self) -> usize {
        self.max_in_flight
    }

    /// Path to the target run bundle directory.
    pub fn run_dir(&self) -> &Path {
        self.inner.run_dir()
    }

    /// Delegate a single JSON-RPC 2.0 line to the strict read-only inspector.
    pub fn handle_line(&self, line: &str) -> Option<String> {
        self.inner.handle_jsonrpc_line(line)
    }

    /// Run synchronous zero-port JSON-RPC 2.0 inspection over `reader`/`writer` (legacy path,
    /// without the `rmcp` protocol layer).
    pub fn serve_stdio<R: BufRead, W: Write>(&self, reader: R, writer: W) -> io::Result<()> {
        self.inner.serve_stdio(reader, writer)
    }

    /// Serve the MCP protocol with the official `rmcp` SDK over process stdin/stdout until the
    /// client disconnects, using [`rmcp::transport::stdio_pipes`] (non-blocking Linux FIFO driver
    /// with cross-platform blocking fallback) and a [`MAX_JSONRPC_LINE_BYTES`] line bound.
    pub async fn serve_rmcp_stdio(self) -> io::Result<()> {
        let (stdin, stdout) = rmcp::transport::stdio_pipes();
        self.serve_rmcp(stdin, stdout).await
    }

    /// Serve the MCP protocol with the official `rmcp` SDK over any async byte stream pair,
    /// dropping incoming lines longer than [`MAX_JSONRPC_LINE_BYTES`] instead of buffering them.
    pub async fn serve_rmcp<R, W>(self, reader: R, writer: W) -> io::Result<()>
    where
        R: AsyncRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        let transport = AsyncRwTransport::<RoleServer, R, W>::new(reader, writer)
            .with_max_line_length(MAX_JSONRPC_LINE_BYTES);
        let running = self
            .serve(transport)
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        match running.waiting().await.map_err(io::Error::other)? {
            QuitReason::JoinError(err) => Err(io::Error::other(err)),
            QuitReason::Closed | QuitReason::Cancelled | _ => Ok(()),
        }
    }

    /// Run one MCP method through the canonical inspector on a bounded blocking worker pool
    /// (bundle verification performs synchronous file I/O) and decode its JSON-RPC result into the
    /// `rmcp` model type. Honors `context.ct` cancellation before acquiring a worker permit and
    /// before launching blocking disk I/O.
    async fn delegate<T>(
        &self,
        method: &'static str,
        params: Value,
        context: RequestContext<RoleServer>,
    ) -> Result<T, ErrorData>
    where
        T: DeserializeOwned + Send + 'static,
    {
        if context.ct.is_cancelled() {
            return Err(ErrorData::internal_error("request cancelled", None));
        }
        let permit = tokio::select! {
            biased;
            _ = context.ct.cancelled() => {
                return Err(ErrorData::internal_error("request cancelled", None));
            }
            res = Arc::clone(&self.in_flight).acquire_owned() => {
                res.map_err(|_| ErrorData::internal_error("inspector concurrency gate closed", None))?
            }
        };
        if context.ct.is_cancelled() {
            drop(permit);
            return Err(ErrorData::internal_error("request cancelled", None));
        }
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let request = json!({"jsonrpc": "2.0", "id": 0, "method": method, "params": params});
            let response = inner
                .handle_jsonrpc_value(&request)
                .ok_or_else(|| ErrorData::internal_error("inspector returned no response", None))?;
            decode_inspector_response(response)
        })
        .await
        .map_err(|e| ErrorData::internal_error(format!("inspector task failed: {e}"), None))?
    }
}

/// Map an inspector JSON-RPC response onto an `rmcp` result or error, preserving its error code.
fn decode_inspector_response<T: DeserializeOwned>(response: Value) -> Result<T, ErrorData> {
    if let Some(error) = response.get("error") {
        let code = error
            .get("code")
            .and_then(Value::as_i64)
            .and_then(|c| i32::try_from(c).ok())
            .unwrap_or(ErrorCode::INTERNAL_ERROR.0);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("inspector error")
            .to_string();
        return Err(ErrorData::new(
            ErrorCode(code),
            message,
            error.get("data").cloned(),
        ));
    }
    let result = response
        .get("result")
        .cloned()
        .ok_or_else(|| ErrorData::internal_error("inspector response missing result", None))?;
    serde_json::from_value(result).map_err(|e| {
        ErrorData::internal_error(
            format!("inspector result does not match the MCP schema: {e}"),
            None,
        )
    })
}

fn to_params<P: serde::Serialize>(params: &P) -> Result<Value, ErrorData> {
    serde_json::to_value(params)
        .map_err(|e| ErrorData::invalid_params(format!("unserializable params: {e}"), None))
}

fn paginated_to_params(request: Option<&PaginatedRequestParams>) -> Result<Value, ErrorData> {
    match request {
        Some(params) => to_params(params),
        None => Ok(json!({})),
    }
}

impl ServerHandler for SwarmTorchMcpAdapterV1 {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(Implementation::new(
            "swarm-torch-mcp",
            env!("CARGO_PKG_VERSION"),
        ))
        .with_instructions(
            "Read-only SwarmTorch run-bundle inspector: every tool and resource call is verified \
             against the bundle's signed manifest, and nothing can be written.",
        )
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        let request = json!({"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}});
        let response = self.inner.handle_jsonrpc_value(&request)?;
        let listed: ListToolsResult = decode_inspector_response(response).ok()?;
        listed.tools.into_iter().find(|t| t.name.as_ref() == name)
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let params = paginated_to_params(request.as_ref())?;
        let mut result: ListToolsResult = self.delegate("tools/list", params, context).await?;
        result.result_type = Some(ResultType::COMPLETE);
        Ok(result)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let mut result: CallToolResult = self
            .delegate("tools/call", to_params(&request)?, context)
            .await?;
        result.result_type = Some(ResultType::COMPLETE);
        Ok(CallToolResponse::Complete(result))
    }

    async fn list_resources(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        let params = paginated_to_params(request.as_ref())?;
        let mut result: ListResourcesResult =
            self.delegate("resources/list", params, context).await?;
        result.result_type = Some(ResultType::COMPLETE);
        Ok(result)
    }

    async fn list_resource_templates(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        let params = paginated_to_params(request.as_ref())?;
        let mut result: ListResourceTemplatesResult = self
            .delegate("resources/templates/list", params, context)
            .await?;
        result.result_type = Some(ResultType::COMPLETE);
        Ok(result)
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        let mut result: ReadResourceResult = self
            .delegate("resources/read", to_params(&request)?, context)
            .await?;
        result.result_type = Some(ResultType::COMPLETE);
        Ok(ReadResourceResponse::Complete(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::service::ServiceError;
    use rmcp::ServiceExt;
    use std::fs;
    use swarm_torch::artifacts::dp_ledger::pre_release_debit_dp_accountant;
    use swarm_torch::artifacts::RunArtifactBundle;
    use swarm_torch::core::crypto::KeyPair;
    use swarm_torch::core::dataops::{
        CacheDecisionV0, DatasetEntryV1, DatasetLineageV1, DatasetRegistryV1, LineageEdgeV1,
        MaterializationRecordV2, MaterializationStatusV0, TransformAuditV0, TrustClass,
        UnsafeReasonV0,
    };
    use swarm_torch::core::dp::{
        DpAdjacencyV1, DpConversionFormulaV1, DpMechanismProfileV1, DpUnitV1,
        PrivacyAccountingClassificationV1, RdpAccountantV1,
    };
    use swarm_torch::core::observe::{AttrMap, MetricRecord, RunId, TraceId};
    use swarm_torch::core::run_graph::{
        node_id_from_key, AssetRefV1, CanonParams, ExecutionTrust, GraphV1, NodeV1, OpKind,
    };

    const JSONRPC_BUNDLE_VERIFICATION_FAILED: i32 = -32001;

    struct TempDirGuard(PathBuf);

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn rich_signed_fixture_bundle(
        tag: &str,
        run_byte: u8,
        key_seed: u8,
    ) -> (TempDirGuard, PathBuf, KeyPair, String) {
        let mut base = std::env::temp_dir();
        base.push(format!(
            "swarmtorch_adapter_mcp_{tag}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let guard = TempDirGuard(base.clone());

        let run_id = RunId::from_bytes([run_byte; 16]);
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
            graph_id: Some(tag.to_string()),
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
                trace_id: TraceId::from_bytes([run_byte; 16]),
                span_id: None,
                name: "dp.round_epsilon".to_string(),
                value: 0.45,
                unit: Some("epsilon".to_string()),
                attrs: AttrMap::new(),
            })
            .unwrap();

        let mut accountant = RdpAccountantV1::new(DpMechanismProfileV1 {
            unit: DpUnitV1::ClientLevel,
            adjacency: DpAdjacencyV1::AddRemoveOne,
            clipping_radius_tau: 1.0,
            noise_multiplier: 1.2,
            target_delta: 1e-5,
            max_epsilon: 3.0,
            conversion_formula: DpConversionFormulaV1::Balle2020Tight,
            classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
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

        let signer = KeyPair::from_seed([key_seed; 32]).unwrap();
        bundle.finalize_signed_manifest(&signer).unwrap();
        let run_dir = bundle.run_dir().to_path_buf();
        let run_id_hex = format!("{run_id}");
        (guard, run_dir, signer, run_id_hex)
    }

    #[test]
    fn mcp_adapter_enforces_strict_verification_over_stdio() {
        let (_guard, run_dir, signer, _run_id_hex) =
            rich_signed_fixture_bundle("stdio", 0x6c, 0x21);

        // Construction refuses LegacyPermissive and empty strict keysets with InvalidInput.
        let legacy_err =
            SwarmTorchMcpAdapterV1::try_new(&run_dir, TrustPolicy::LegacyPermissive).unwrap_err();
        assert_eq!(legacy_err.kind(), io::ErrorKind::InvalidInput);
        let empty_err = SwarmTorchMcpAdapterV1::try_new(
            &run_dir,
            TrustPolicy::Strict {
                trusted_public_keys: vec![],
                expected_run_id: None,
                min_signed_unix_nanos: None,
            },
        )
        .unwrap_err();
        assert_eq!(empty_err.kind(), io::ErrorKind::InvalidInput);

        // Semaphore permit bounds clamp safely on both extremes (0 -> 1, usize::MAX -> MAX_PERMITS).
        let adapter_max = SwarmTorchMcpAdapterV1::strict_with_key(&run_dir, *signer.public_key())
            .unwrap()
            .with_max_in_flight_requests(usize::MAX);
        assert_eq!(adapter_max.max_in_flight_requests(), Semaphore::MAX_PERMITS);

        let adapter = adapter_max.with_max_in_flight_requests(0);
        assert_eq!(adapter.max_in_flight_requests(), 1);
        assert_eq!(adapter.run_dir(), run_dir.as_path());

        let resp = adapter
            .handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
            .unwrap();
        assert!(resp.contains("inspect_run_report"));
        assert!(resp.contains("trace_aggregation_budget"));

        let input_stream = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"trace_aggregation_budget\",\"arguments\":{}}}\n"
        );
        let mut output_buf = Vec::new();
        adapter
            .serve_stdio(
                std::io::Cursor::new(input_stream.as_bytes()),
                &mut output_buf,
            )
            .unwrap();
        let output_text = String::from_utf8(output_buf).unwrap();
        let lines: Vec<&str> = output_text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("protocolVersion"));
        assert!(lines[1].contains("trace_aggregation_budget"));
        assert!(lines[2].contains("experimental_hypothetical_budget_only"));
    }

    #[test]
    fn mcp_adapter_child_subprocess_stdio_pipe_roundtrip() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        if let Ok(run_dir) = std::env::var("SWARMTORCH_MCP_TEST_CHILD_RUN_DIR") {
            let key_byte: u8 = std::env::var("SWARMTORCH_MCP_TEST_CHILD_KEY_BYTE")
                .unwrap()
                .parse()
                .unwrap();
            let signer = KeyPair::from_seed([key_byte; 32]).unwrap();
            let adapter =
                SwarmTorchMcpAdapterV1::strict_with_key(run_dir, *signer.public_key()).unwrap();
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            let mut out_lock = stdout.lock();
            out_lock.write_all(b"\n").unwrap();
            out_lock.flush().unwrap();
            adapter.serve_stdio(stdin.lock(), out_lock).unwrap();
            return;
        }

        let (_guard, run_dir, _signer, _run_id_hex) =
            rich_signed_fixture_bundle("subproc_legacy", 0x7d, 0x42);

        let exe = std::env::current_exe().unwrap();
        let mut child = Command::new(exe)
            .arg("mcp_adapter_child_subprocess_stdio_pipe_roundtrip")
            .arg("--nocapture")
            .env("SWARMTORCH_MCP_TEST_CHILD_RUN_DIR", &run_dir)
            .env("SWARMTORCH_MCP_TEST_CHILD_KEY_BYTE", "66")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        {
            let stdin = child.stdin.as_mut().unwrap();
            stdin
                .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"tools/list\"}\n")
                .unwrap();
            stdin
                .write_all(
                    b"{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{\"name\":\"trace_aggregation_budget\",\"arguments\":{}}}\n",
                )
                .unwrap();
            stdin.flush().unwrap();
        }
        drop(child.stdin.take());

        let out = child.wait_with_output().unwrap();
        let stdout_str = String::from_utf8_lossy(&out.stdout);
        let stderr_str = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "child failed: stdout={stdout_str}, stderr={stderr_str}"
        );
        let jsonrpc_lines: Vec<&str> = stdout_str
            .lines()
            .filter(|l| l.starts_with('{') && l.contains("\"jsonrpc\":\"2.0\""))
            .collect();
        assert_eq!(
            jsonrpc_lines.len(),
            2,
            "unexpected child stdout={stdout_str:?}, stderr={stderr_str:?}"
        );
        assert!(jsonrpc_lines[0].contains("\"id\":10"));
        assert!(jsonrpc_lines[0].contains("inspect_run_report"));
        assert!(jsonrpc_lines[1].contains("\"id\":11"));
        assert!(jsonrpc_lines[1].contains("experimental_hypothetical_budget_only"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rmcp_client_exercises_all_tools_resources_and_strict_error_codes() {
        let (_guard, run_dir, signer, run_id_hex) = rich_signed_fixture_bundle("rmcp", 0x5e, 0x33);
        let adapter =
            SwarmTorchMcpAdapterV1::strict_with_key(&run_dir, *signer.public_key()).unwrap();

        // `ServerHandler::get_tool` resolves canonical tool metadata and rejects unknown tools.
        for name in [
            "inspect_run_report",
            "audit_graph_trust_and_cycles",
            "predict_node_outputs",
            "trace_aggregation_budget",
        ] {
            let tool = adapter
                .get_tool(name)
                .unwrap_or_else(|| panic!("missing tool {name}"));
            assert_eq!(tool.name.as_ref(), name);
            let ann = tool.annotations.expect("read-only tool annotations");
            assert_eq!(ann.read_only_hint, Some(true));
            assert_eq!(ann.destructive_hint, Some(false));
        }
        assert!(adapter.get_tool("mutate_bundle").is_none());

        let (server_io, client_io) = tokio::io::duplex(1 << 20);
        let (server_read, server_write) = tokio::io::split(server_io);
        let server = tokio::spawn(adapter.serve_rmcp(server_read, server_write));
        let client = ().serve(client_io).await.unwrap();

        // Negotiated peer info & strict capability surface.
        let info = client.peer_info().unwrap();
        assert_eq!(
            info.server_info.as_ref().map(|i| i.name.as_str()),
            Some("swarm-torch-mcp")
        );
        assert!(info.capabilities.tools.is_some());
        assert!(info.capabilities.resources.is_some());
        assert!(info.capabilities.prompts.is_none());
        assert!(info.capabilities.completions.is_none());

        // List all 4 canonical read-only tools.
        let tools = client.list_tools(None).await.unwrap();
        let names: Vec<String> = tools.tools.iter().map(|t| t.name.to_string()).collect();
        assert_eq!(
            names,
            vec![
                "inspect_run_report",
                "audit_graph_trust_and_cycles",
                "predict_node_outputs",
                "trace_aggregation_budget",
            ]
        );
        for tool in &tools.tools {
            let ann = tool.annotations.as_ref().expect("tool annotations");
            assert_eq!(ann.read_only_hint, Some(true));
            assert_eq!(ann.destructive_hint, Some(false));
        }

        // Exercise all 4 tools and verify structured output fidelity.
        let inspect = client
            .call_tool(CallToolRequestParams::new("inspect_run_report"))
            .await
            .unwrap();
        let inspect_sc = inspect.structured_content.as_ref().unwrap();
        assert_eq!(inspect_sc["verified_strict"], true);
        assert_eq!(inspect_sc["node_count"], 2);
        assert_eq!(inspect_sc["materialization_count"], 1);

        let audit = client
            .call_tool(CallToolRequestParams::new("audit_graph_trust_and_cycles"))
            .await
            .unwrap();
        let audit_sc = audit.structured_content.as_ref().unwrap();
        assert_eq!(audit_sc["cycle_free"], true);
        assert_eq!(
            audit_sc["sandboxed_extension_nodes"],
            json!(["ext/normalize"])
        );
        assert_eq!(audit_sc["untrusted_datasets"], json!(["dataset://ns/raw"]));

        let mut predict_args = serde_json::Map::new();
        predict_args.insert("node_key".to_string(), json!("ext/normalize"));
        let mut predict_req = CallToolRequestParams::new("predict_node_outputs");
        predict_req.arguments = Some(predict_args);
        let predict = client.call_tool(predict_req).await.unwrap();
        let predict_sc = predict.structured_content.as_ref().unwrap();
        assert_eq!(predict_sc["node_count"], 1);
        assert_eq!(
            predict_sc["predictions"][0]["predicted_outputs"][0]["observed_rows"],
            42
        );

        let budget = client
            .call_tool(CallToolRequestParams::new("trace_aggregation_budget"))
            .await
            .unwrap();
        let budget_sc = budget.structured_content.as_ref().unwrap();
        assert_eq!(budget_sc["guarantees_differential_privacy"], false);
        assert_eq!(
            budget_sc["accounting_classification"],
            "experimental_hypothetical_budget_only"
        );
        assert_eq!(budget_sc["committed_rounds"], 1);

        // List and read all 4 verified resources.
        let resources = client.list_resources(None).await.unwrap();
        assert_eq!(resources.resources.len(), 4);
        for res in &resources.resources {
            let read = client
                .read_resource(ReadResourceRequestParams::new(res.uri.clone()))
                .await
                .unwrap();
            assert_eq!(read.contents.len(), 1);
            match &read.contents[0] {
                rmcp::model::ResourceContents::TextResourceContents {
                    uri,
                    mime_type,
                    text,
                    ..
                } => {
                    assert_eq!(uri, &res.uri);
                    assert_eq!(mime_type.as_deref(), Some("application/json"));
                    let _: Value = serde_json::from_str(text).unwrap();
                }
                other => panic!("unexpected resource content variant: {other:?}"),
            }
        }

        // Strict negative assertions: exact JSON-RPC error codes, never loose `Err(_)`.
        let unknown_tool = client
            .call_tool(CallToolRequestParams::new("mutate_bundle"))
            .await;
        assert!(
            matches!(
                unknown_tool,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::INVALID_PARAMS
            ),
            "expected INVALID_PARAMS (-32602), got {unknown_tool:?}"
        );

        // Extra undeclared argument violates `additionalProperties: false`.
        let mut extra_args = serde_json::Map::new();
        extra_args.insert("write".to_string(), json!(true));
        let mut extra_req = CallToolRequestParams::new("inspect_run_report");
        extra_req.arguments = Some(extra_args);
        let extra_res = client.call_tool(extra_req).await;
        assert!(
            matches!(
                extra_res,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::INVALID_PARAMS
            ),
            "expected INVALID_PARAMS (-32602), got {extra_res:?}"
        );

        // Path traversal & cross-run URI rejected with INVALID_PARAMS (-32602).
        for bad_uri in [
            format!("swarmtorch://runs/{run_id_hex}/../../etc/passwd"),
            format!("swarmtorch://runs/{}/manifest", "ff".repeat(16)),
            "file:///etc/passwd".to_string(),
        ] {
            let bad_read = client
                .read_resource(ReadResourceRequestParams::new(bad_uri.clone()))
                .await;
            assert!(
                matches!(
                    bad_read,
                    Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::INVALID_PARAMS
                ),
                "expected INVALID_PARAMS for {bad_uri}, got {bad_read:?}"
            );
        }

        // Resource templates surface returns empty list on verified bundle and rejects cursors.
        let templates = client.list_resource_templates(None).await.unwrap();
        assert!(templates.resource_templates.is_empty());

        // Unsupported pagination cursors rejected with INVALID_PARAMS (-32602).
        let bad_cursor_tools = client
            .list_tools(Some(
                PaginatedRequestParams::default().with_cursor(Some("p2".to_string())),
            ))
            .await;
        assert!(
            matches!(
                bad_cursor_tools,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::INVALID_PARAMS
            ),
            "expected INVALID_PARAMS for tools/list cursor, got {bad_cursor_tools:?}"
        );
        let bad_cursor_res = client
            .list_resources(Some(
                PaginatedRequestParams::default().with_cursor(Some("p2".to_string())),
            ))
            .await;
        assert!(
            matches!(
                bad_cursor_res,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::INVALID_PARAMS
            ),
            "expected INVALID_PARAMS for resources/list cursor, got {bad_cursor_res:?}"
        );
        let bad_cursor_tpl = client
            .list_resource_templates(Some(
                PaginatedRequestParams::default().with_cursor(Some("p2".to_string())),
            ))
            .await;
        assert!(
            matches!(
                bad_cursor_tpl,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::INVALID_PARAMS
            ),
            "expected INVALID_PARAMS for resources/templates/list cursor, got {bad_cursor_tpl:?}"
        );

        // Unadvertised prompts surface fails closed with METHOD_NOT_FOUND (-32601).
        let bad_prompts = client.list_prompts(None).await;
        assert!(
            matches!(
                bad_prompts,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::METHOD_NOT_FOUND
            ),
            "expected METHOD_NOT_FOUND (-32601), got {bad_prompts:?}"
        );

        // Tamper a signed artifact on disk mid-session -> every bundle-reading call fails closed with -32001.
        fs::write(run_dir.join("metrics.ndjson"), b"{\"tampered\":true}\n").unwrap();
        let tampered_list = client.list_resources(None).await;
        assert!(
            matches!(
                tampered_list,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode(JSONRPC_BUNDLE_VERIFICATION_FAILED)
            ),
            "expected -32001 on tampered bundle, got {tampered_list:?}"
        );
        let tampered_tpl = client.list_resource_templates(None).await;
        assert!(
            matches!(
                tampered_tpl,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode(JSONRPC_BUNDLE_VERIFICATION_FAILED)
            ),
            "expected -32001 on tampered bundle resource templates, got {tampered_tpl:?}"
        );
        let tampered_tool = client
            .call_tool(CallToolRequestParams::new("inspect_run_report"))
            .await;
        assert!(
            matches!(
                tampered_tool,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode(JSONRPC_BUNDLE_VERIFICATION_FAILED)
            ),
            "expected -32001 on tampered bundle tool call, got {tampered_tool:?}"
        );

        client.cancel().await.unwrap();
        server.await.unwrap().unwrap();

        // Wrong Ed25519 key also fails closed with -32001 on a fresh un-tampered bundle.
        let (_guard2, run_dir2, _signer2, _run_id_hex2) =
            rich_signed_fixture_bundle("rmcp_wrong_key", 0x60, 0x35);
        let wrong_signer = KeyPair::from_seed([0x99; 32]).unwrap();
        let bad_adapter =
            SwarmTorchMcpAdapterV1::strict_with_key(&run_dir2, *wrong_signer.public_key()).unwrap();
        let (s_io, c_io) = tokio::io::duplex(1 << 20);
        let (s_r, s_w) = tokio::io::split(s_io);
        let bad_server = tokio::spawn(bad_adapter.serve_rmcp(s_r, s_w));
        let bad_client = ().serve(c_io).await.unwrap();
        let wrong_key_err = bad_client
            .call_tool(CallToolRequestParams::new("inspect_run_report"))
            .await;
        assert!(
            matches!(
                wrong_key_err,
                Err(ServiceError::McpError(ref e)) if e.code == ErrorCode(JSONRPC_BUNDLE_VERIFICATION_FAILED)
            ),
            "expected -32001 for wrong public key, got {wrong_key_err:?}"
        );
        bad_client.cancel().await.unwrap();
        bad_server.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rmcp_transport_enforces_line_bound_and_protocol_version_wire_shapes() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::time::{timeout, Duration};

        let (_guard, run_dir, signer, run_id_hex) =
            rich_signed_fixture_bundle("rmcp_bound", 0x5f, 0x34);
        let adapter =
            SwarmTorchMcpAdapterV1::strict_with_key(&run_dir, *signer.public_key()).unwrap();
        let (server_io, client_io) = tokio::io::duplex(4 << 20);
        let (server_read, server_write) = tokio::io::split(server_io);
        let server = tokio::spawn(adapter.serve_rmcp(server_read, server_write));
        let (client_read, mut client_write) = tokio::io::split(client_io);
        let mut lines = BufReader::new(client_read).lines();

        // 1. Legacy 2025-06-18 handshake + tools/list: `resultType` must be stripped for legacy peer.
        client_write
            .write_all(
                concat!(
                    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","#,
                    r#""capabilities":{},"clientInfo":{"name":"bound-test","version":"0"}}}"#,
                    "\n",
                    r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
                    "\n",
                    r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#,
                    "\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let init = timeout(Duration::from_secs(10), lines.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let init_val: Value = serde_json::from_str(&init).unwrap();
        assert_eq!(init_val["id"], 1);
        assert_eq!(init_val["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(init_val["result"]["serverInfo"]["name"], "swarm-torch-mcp");

        let tools_line = timeout(Duration::from_secs(10), lines.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let tools_val: Value = serde_json::from_str(&tools_line).unwrap();
        assert_eq!(tools_val["id"], 2);
        assert!(tools_val["result"].get("resultType").is_none());

        // 2. Oversized line (> 1 MiB) is dropped without executing id=99, and id=100 ping succeeds.
        let pad = "x".repeat(MAX_JSONRPC_LINE_BYTES);
        let oversized = format!(
            r#"{{"jsonrpc":"2.0","id":99,"method":"tools/list","params":{{"_meta":{{"pad":"{pad}"}}}}}}"#
        );
        client_write.write_all(oversized.as_bytes()).await.unwrap();
        client_write.write_all(b"\n").await.unwrap();
        client_write
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":100,\"method\":\"ping\"}\n")
            .await
            .unwrap();

        let ping_line = timeout(Duration::from_secs(10), lines.next_line())
            .await
            .expect("server stopped answering after an oversized line")
            .unwrap()
            .expect("server closed stream after an oversized line");
        let ping_val: Value = serde_json::from_str(&ping_line).unwrap();
        assert_eq!(
            ping_val["id"], 100,
            "oversized id=99 must be dropped; saw {ping_val}"
        );
        assert!(ping_val["result"].is_object());

        // 3. Unterminated oversized flood followed by EOF must terminate cleanly without OOM/panic.
        let unterminated = vec![b'z'; MAX_JSONRPC_LINE_BYTES + 4096];
        client_write.write_all(&unterminated).await.unwrap();
        client_write.shutdown().await.unwrap();
        drop(client_write);
        drop(lines);
        let server_res = timeout(Duration::from_secs(10), server)
            .await
            .expect("server hung after unterminated oversized EOF")
            .unwrap();
        assert!(
            server_res.is_ok(),
            "server errored on clean EOF: {server_res:?}"
        );

        // 4. Protocol 2026-07-28 wire check across tools/list, resources/list,
        // resources/templates/list, resources/read, and tools/call.
        let adapter_v26 =
            SwarmTorchMcpAdapterV1::strict_with_key(&run_dir, *signer.public_key()).unwrap();
        let (s26_io, c26_io) = tokio::io::duplex(1 << 20);
        let (s26_r, s26_w) = tokio::io::split(s26_io);
        let server26 = tokio::spawn(adapter_v26.serve_rmcp(s26_r, s26_w));
        let (c26_r, mut c26_w) = tokio::io::split(c26_io);
        let mut lines26 = BufReader::new(c26_r).lines();
        let meta = r#""_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}"#;
        let v26_requests = [
            format!(r#"{{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{{{meta}}}}}"#),
            format!(r#"{{"jsonrpc":"2.0","id":2,"method":"resources/list","params":{{{meta}}}}}"#),
            format!(
                r#"{{"jsonrpc":"2.0","id":3,"method":"resources/templates/list","params":{{{meta}}}}}"#
            ),
            format!(
                r#"{{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{{{meta},"uri":"swarmtorch://runs/{run_id_hex}/manifest"}}}}"#
            ),
            format!(
                r#"{{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{{{meta},"name":"inspect_run_report","arguments":{{}}}}}}"#
            ),
        ];
        for (idx, req) in v26_requests.iter().enumerate() {
            c26_w.write_all(req.as_bytes()).await.unwrap();
            c26_w.write_all(b"\n").await.unwrap();
            let line = timeout(Duration::from_secs(10), lines26.next_line())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let val: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(val["id"], (idx + 1) as u64, "unexpected id in {val}");
            assert_eq!(
                val["result"]["resultType"], "complete",
                "missing resultType in {val}"
            );
            if idx < 4 {
                // Cacheable read/list methods carry 2026-07-28 cache hints.
                assert_eq!(val["result"]["ttlMs"], 0, "missing ttlMs in {val}");
                assert_eq!(
                    val["result"]["cacheScope"], "private",
                    "missing cacheScope in {val}"
                );
            }
        }
        c26_w.shutdown().await.unwrap();
        drop(c26_w);
        drop(lines26);
        timeout(Duration::from_secs(10), server26)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rmcp_concurrent_pipelined_requests_respect_semaphore_bound_and_cancellation() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::time::{timeout, Duration};

        let (_guard, run_dir, signer, _run_id_hex) =
            rich_signed_fixture_bundle("rmcp_conc", 0x61, 0x36);
        let adapter = SwarmTorchMcpAdapterV1::strict_with_key(&run_dir, *signer.public_key())
            .unwrap()
            .with_max_in_flight_requests(2);
        let (server_io, client_io) = tokio::io::duplex(1 << 20);
        let (server_read, server_write) = tokio::io::split(server_io);
        let server = tokio::spawn(adapter.serve_rmcp(server_read, server_write));
        let client = ().serve(client_io).await.unwrap();
        let peer = client.peer().clone();

        let mut set = tokio::task::JoinSet::new();
        for _ in 0..12 {
            let p = peer.clone();
            set.spawn(async move {
                let res = p
                    .call_tool(CallToolRequestParams::new("inspect_run_report"))
                    .await
                    .unwrap();
                assert_eq!(res.structured_content.unwrap()["node_count"], 2);
            });
        }
        while let Some(joined) = set.join_next().await {
            joined.unwrap();
        }
        client.cancel().await.unwrap();
        server.await.unwrap().unwrap();

        // Real wire cancellation while the single semaphore permit is held:
        // request id=77 blocks waiting on the permit, receives `notifications/cancelled`,
        // aborts promptly without launching blocking I/O or leaking permits, and id=78 succeeds
        // once the permit is released.
        let cancel_adapter =
            SwarmTorchMcpAdapterV1::strict_with_key(&run_dir, *signer.public_key())
                .unwrap()
                .with_max_in_flight_requests(1);
        let sem = Arc::clone(&cancel_adapter.in_flight);
        let held_permit = Arc::clone(&sem).acquire_owned().await.unwrap();
        assert_eq!(sem.available_permits(), 0);

        let (s_io, c_io) = tokio::io::duplex(1 << 20);
        let (s_r, s_w) = tokio::io::split(s_io);
        let cancel_server = tokio::spawn(cancel_adapter.serve_rmcp(s_r, s_w));
        let (c_r, mut c_w) = tokio::io::split(c_io);
        let mut c_lines = BufReader::new(c_r).lines();

        c_w.write_all(
            concat!(
                r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"cancel-test","version":"0"}}}"#,
                "\n",
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
                "\n",
                r#"{"jsonrpc":"2.0","id":77,"method":"tools/call","params":{"name":"inspect_run_report","arguments":{}}}"#,
                "\n",
            )
            .as_bytes(),
        )
        .await
        .unwrap();

        let init_line = timeout(Duration::from_secs(5), c_lines.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(init_line.contains("\"id\":1"), "{init_line}");

        // Yield briefly so id=77 enters the semaphore `select!`, then cancel it over the wire.
        tokio::time::sleep(Duration::from_millis(50)).await;
        c_w.write_all(
            concat!(
                r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":77,"reason":"client timeout"}}"#,
                "\n",
            )
            .as_bytes(),
        )
        .await
        .unwrap();

        // Give the cancelled handler task a moment to abort cleanly while the permit is still held.
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(held_permit);
        assert_eq!(
            sem.available_permits(),
            1,
            "cancelled waiter must not leak semaphore permits"
        );

        // Subsequent request id=78 must execute cleanly with zero leaked permits.
        c_w.write_all(
            concat!(
                r#"{"jsonrpc":"2.0","id":78,"method":"tools/call","params":{"name":"inspect_run_report","arguments":{}}}"#,
                "\n",
            )
            .as_bytes(),
        )
        .await
        .unwrap();

        let mut saw_78 = false;
        while let Ok(Some(line)) = timeout(Duration::from_secs(5), c_lines.next_line())
            .await
            .expect("server hung after cancellation")
        {
            let v: Value = serde_json::from_str(&line).unwrap();
            if v["id"] == 77 {
                // If rmcp forwards the cancellation error response for id=77, it must NOT be Ok result.
                assert!(
                    v.get("result").is_none(),
                    "cancelled request id=77 must not return a result: {v}"
                );
            }
            if v["id"] == 78 {
                assert_eq!(v["result"]["structuredContent"]["node_count"], 2);
                saw_78 = true;
                break;
            }
        }
        assert!(saw_78, "subsequent request id=78 never completed");
        assert_eq!(sem.available_permits(), 1);

        c_w.shutdown().await.unwrap();
        drop(c_w);
        drop(c_lines);
        timeout(Duration::from_secs(5), cancel_server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}
