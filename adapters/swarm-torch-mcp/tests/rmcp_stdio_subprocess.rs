//! End-to-end OS pipe subprocess test exercising [`SwarmTorchMcpAdapterV1::serve_rmcp_stdio`]
//! (`rmcp::transport::stdio_pipes`) with a real `rmcp` client on unadulterated process stdin/stdout.
//!
//! Uses `harness = false` so Rust's `libtest` never pollutes the child process's stdout JSON-RPC stream.

use std::fs;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use rmcp::model::{CallToolRequestParams, ErrorCode, ReadResourceRequestParams};
use rmcp::service::ServiceError;
use rmcp::transport::async_rw::AsyncRwTransport;
use rmcp::{RoleClient, ServiceExt};
use swarm_torch::artifacts::RunArtifactBundle;
use swarm_torch::core::crypto::KeyPair;
use swarm_torch::core::dataops::{DatasetLineageV1, DatasetRegistryV1};
use swarm_torch::core::observe::RunId;
use swarm_torch::core::run_graph::GraphV1;
use swarm_torch_mcp::SwarmTorchMcpAdapterV1;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(async_main());
}

async fn async_main() {
    if let Ok(run_dir) = std::env::var("SWARMTORCH_RMCP_SUBPROC_RUN_DIR") {
        let key_byte: u8 = std::env::var("SWARMTORCH_RMCP_SUBPROC_KEY_BYTE")
            .expect("child key byte env")
            .parse()
            .expect("parse child key byte");
        let signer = KeyPair::from_seed([key_byte; 32]).expect("child keypair");
        let adapter = SwarmTorchMcpAdapterV1::strict_with_key(run_dir, *signer.public_key())
            .expect("init strict adapter");
        adapter
            .serve_rmcp_stdio()
            .await
            .expect("serve_rmcp_stdio exited with error");
        return;
    }

    let mut base = std::env::temp_dir();
    base.push(format!(
        "swarmtorch_rmcp_subproc_os_pipe_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).expect("create temp dir");
    let _temp_guard = TempDirGuard(base.clone());

    let run_id = RunId::from_bytes([0x62; 16]);
    let bundle = RunArtifactBundle::create(&base, run_id).expect("create bundle");
    bundle
        .write_graph(&GraphV1 {
            schema_version: 1,
            graph_id: Some("rmcp-os-pipe-test".to_string()),
            nodes: vec![],
            edges: vec![],
        })
        .expect("write graph");
    bundle
        .write_dataset_registry(&DatasetRegistryV1::default())
        .expect("write registry");
    bundle
        .write_dataset_lineage(&DatasetLineageV1::default())
        .expect("write lineage");
    let signer = KeyPair::from_seed([0x37; 32]).expect("signer");
    bundle
        .finalize_signed_manifest(&signer)
        .expect("finalize signed manifest");
    let run_dir: PathBuf = bundle.run_dir().to_path_buf();
    let run_id_hex = format!("{run_id}");

    let exe = std::env::current_exe().expect("current_exe");
    let mut child = Command::new(exe)
        .env("SWARMTORCH_RMCP_SUBPROC_RUN_DIR", &run_dir)
        .env("SWARMTORCH_RMCP_SUBPROC_KEY_BYTE", "55")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn rmcp stdio child");

    let child_stdin = child.stdin.take().expect("child stdin");
    let child_stdout = child.stdout.take().expect("child stdout");
    let mut child_stderr = child.stderr.take().expect("child stderr");
    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::new();
        let _ = child_stderr.read_to_end(&mut buf).await;
        String::from_utf8_lossy(&buf).into_owned()
    });

    let rpc_timeout = Duration::from_secs(15);
    let transport = AsyncRwTransport::<RoleClient, _, _>::new_client(child_stdout, child_stdin);
    let client = timeout(rpc_timeout, ().serve(transport))
        .await
        .expect("rmcp client handshake timed out")
        .expect("rmcp client handshake failed");

    let info = client.peer_info().expect("peer info");
    assert_eq!(
        info.server_info.as_ref().map(|i| i.name.as_str()),
        Some("swarm-torch-mcp")
    );
    assert!(info.capabilities.tools.is_some());
    assert!(info.capabilities.resources.is_some());

    let tools = timeout(rpc_timeout, client.list_tools(None))
        .await
        .expect("list_tools timed out")
        .expect("list_tools failed");
    assert_eq!(tools.tools.len(), 4);

    let templates = timeout(rpc_timeout, client.list_resource_templates(None))
        .await
        .expect("list_resource_templates timed out")
        .expect("list_resource_templates failed");
    assert!(templates.resource_templates.is_empty());

    let budget = timeout(
        rpc_timeout,
        client.call_tool(CallToolRequestParams::new("trace_aggregation_budget")),
    )
    .await
    .expect("trace_aggregation_budget timed out")
    .expect("trace_aggregation_budget failed");
    assert_eq!(
        budget.structured_content.expect("structuredContent")["accounting_classification"],
        "experimental_hypothetical_budget_only"
    );

    let manifest = timeout(
        rpc_timeout,
        client.read_resource(ReadResourceRequestParams::new(format!(
            "swarmtorch://runs/{run_id_hex}/manifest"
        ))),
    )
    .await
    .expect("read manifest timed out")
    .expect("read manifest resource failed");
    assert_eq!(manifest.contents.len(), 1);

    let refused = timeout(
        rpc_timeout,
        client.call_tool(CallToolRequestParams::new("mutate_bundle")),
    )
    .await
    .expect("mutate_bundle timed out");
    assert!(
        matches!(
            refused,
            Err(ServiceError::McpError(ref e)) if e.code == ErrorCode::INVALID_PARAMS
        ),
        "expected INVALID_PARAMS (-32602), got {refused:?}"
    );

    // Tamper bundle mid-session -> OS child fails closed with -32001.
    fs::write(run_dir.join("graph.json"), b"{\"tampered\":true}").expect("tamper graph.json");
    let tampered = timeout(
        rpc_timeout,
        client.call_tool(CallToolRequestParams::new("inspect_run_report")),
    )
    .await
    .expect("tampered inspect_run_report timed out");
    assert!(
        matches!(
            tampered,
            Err(ServiceError::McpError(ref e)) if e.code == ErrorCode(-32001)
        ),
        "expected -32001 on tampered bundle over OS pipe, got {tampered:?}"
    );

    timeout(rpc_timeout, client.cancel())
        .await
        .expect("client cancel timed out")
        .expect("client cancel failed");
    let status = timeout(Duration::from_secs(10), child.wait())
        .await
        .expect("rmcp stdio child did not exit cleanly after stdin EOF")
        .expect("wait child");
    let stderr_out = timeout(Duration::from_secs(5), stderr_task)
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();
    assert!(
        status.success(),
        "rmcp stdio child exited with {status:?}; stderr={stderr_out}"
    );

    println!("ok   rmcp_stdio_subprocess: OS stdin/stdout pipe roundtrip + tamper fail-closed");
}
