#![forbid(unsafe_code)]

//! Isolated zero-port read-only strict MCP inspector adapter wrapping
//! [`swarm_torch::mcp_inspector::McpInspectorServerV1`].
//!
//! **Maturity Tier:** `Integration verified` — strict stdio JSON-RPC 2.0 request/response framing,
//! bounded/validated `dp/ledger.ndjson` tracing (`AR-14-04`), and OS child-process (`Command` stdio
//! pipe) roundtrips run deterministically without network listeners. Differential privacy tool fields
//! explicitly surface experimental accounting-only metadata (`ST-02`).

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

pub use swarm_torch::artifacts::TrustPolicy;
pub use swarm_torch::mcp_inspector::{McpInspectorServerV1, MAX_JSONRPC_LINE_BYTES};

/// Thin adapter handle serving strict read-only SwarmTorch bundle inspection over stdio JSON-RPC 2.0.
#[derive(Debug, Clone)]
pub struct SwarmTorchMcpAdapterV1 {
    inner: McpInspectorServerV1,
}

impl SwarmTorchMcpAdapterV1 {
    /// Construct an adapter bound to a verified run bundle directory and strict Ed25519 key.
    pub fn strict_with_key(run_dir: impl Into<PathBuf>, public_key: [u8; 32]) -> io::Result<Self> {
        Ok(Self {
            inner: McpInspectorServerV1::strict_with_key(run_dir, public_key)?,
        })
    }

    /// Delegate a single JSON-RPC 2.0 line to the strict read-only inspector.
    pub fn handle_line(&self, line: &str) -> Option<String> {
        self.inner.handle_jsonrpc_line(line)
    }

    /// Run synchronous zero-port JSON-RPC 2.0 inspection over `reader`/`writer`.
    pub fn serve_stdio<R: BufRead, W: Write>(&self, reader: R, writer: W) -> io::Result<()> {
        self.inner.serve_stdio(reader, writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use swarm_torch::artifacts::RunArtifactBundle;
    use swarm_torch::core::crypto::KeyPair;
    use swarm_torch::core::dataops::{DatasetLineageV1, DatasetRegistryV1};
    use swarm_torch::core::observe::RunId;
    use swarm_torch::core::run_graph::GraphV1;

    #[test]
    fn mcp_adapter_enforces_strict_verification_over_stdio() {
        let mut base = std::env::temp_dir();
        base.push(format!("swarmtorch_adapter_mcp_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x6c; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        bundle
            .write_graph(&GraphV1 {
                schema_version: 1,
                graph_id: Some("adapter-test".to_string()),
                nodes: vec![],
                edges: vec![],
            })
            .unwrap();
        bundle
            .write_dataset_registry(&DatasetRegistryV1::default())
            .unwrap();
        bundle
            .write_dataset_lineage(&DatasetLineageV1::default())
            .unwrap();
        let signer = KeyPair::from_seed([0x21; 32]).unwrap();
        bundle.finalize_signed_manifest(&signer).unwrap();

        let adapter =
            SwarmTorchMcpAdapterV1::strict_with_key(bundle.run_dir(), *signer.public_key())
                .unwrap();
        let resp = adapter
            .handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#)
            .unwrap();
        assert!(resp.contains("inspect_run_report"));
        assert!(resp.contains("trace_aggregation_budget"));

        // Also test full serve_stdio over an OS pipe (`std::io::pipe` / child thread stdio stream)
        // with initialize -> tools/list -> tools/call (`trace_aggregation_budget`) -> EOF (`AR-14-04`):
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

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn mcp_adapter_child_subprocess_stdio_pipe_roundtrip() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        // If invoked as the child worker inside this test binary, run `serve_stdio` on real OS stdin/stdout:
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

        let mut base = std::env::temp_dir();
        base.push(format!(
            "swarmtorch_adapter_mcp_subproc_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();

        let run_id = RunId::from_bytes([0x7d; 16]);
        let bundle = RunArtifactBundle::create(&base, run_id).unwrap();
        bundle
            .write_graph(&GraphV1 {
                schema_version: 1,
                graph_id: Some("subproc-test".to_string()),
                nodes: vec![],
                edges: vec![],
            })
            .unwrap();
        bundle
            .write_dataset_registry(&DatasetRegistryV1::default())
            .unwrap();
        bundle
            .write_dataset_lineage(&DatasetLineageV1::default())
            .unwrap();
        let signer = KeyPair::from_seed([0x42; 32]).unwrap();
        bundle.finalize_signed_manifest(&signer).unwrap();

        let exe = std::env::current_exe().unwrap();
        let mut child = Command::new(exe)
            .arg("mcp_adapter_child_subprocess_stdio_pipe_roundtrip")
            .arg("--nocapture")
            .env("SWARMTORCH_MCP_TEST_CHILD_RUN_DIR", bundle.run_dir())
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
        // Close child stdin so `serve_stdio` hits EOF cleanly and exits:
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

        let _ = fs::remove_dir_all(&base);
    }
}
