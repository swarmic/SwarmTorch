#![forbid(unsafe_code)]

//! Isolated zero-port read-only strict MCP inspector adapter wrapping
//! [`swarm_torch::mcp_inspector::McpInspectorServerV1`].
//!
//! **Maturity Tier:** `Simulation verified` — strict stdio JSON-RPC 2.0 request/response framing
//! and read-only bundle verification run deterministically without network listeners. Differential
//! privacy tool fields explicitly surface experimental accounting-only metadata (`ST-02`).

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

        let _ = fs::remove_dir_all(&base);
    }
}
