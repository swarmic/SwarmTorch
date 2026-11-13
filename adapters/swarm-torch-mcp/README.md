# swarm-torch-mcp

Isolated zero-port read-only strict Model Context Protocol (MCP) adapter wrapping `swarm_torch::mcp_inspector::McpInspectorServerV1`.

- Operates strictly over JSON-RPC 2.0 `stdio` (zero HTTP/TCP ports).
- Fails closed unless configured with `TrustPolicy::Strict` and valid Ed25519 signed run bundles.
- Kept isolated from the core workspace `Cargo.lock` until external MCP SDK crates are introduced and audited.
