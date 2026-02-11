//! SwarmTorch run graph schema (artifact spine).
//!
//! This module defines the executable `graph.json` schema described in ADR-0017.
//! The goal is:
//! - semantics-first DAG representation (data + training + comms + governance)
//! - stable node identity derived from a human-stable `node_key`
//! - deterministic `node_def_hash` for caching/materialization correctness
//!
//! IMPORTANT: Do not hash JSON bytes for identity. Hash canonical binary (postcard)
//! encoding of a canonical struct (`NodeDefCanonicalV1`).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use sha2::{Digest, Sha256};

use crate::observe::TraceId;

/// Graph schema version for `graph.json`.
pub const GRAPH_SCHEMA_V1: u32 = 1;

/// A stable node id used for addressing (derived from `node_key`).
///
/// Encoding: 16 bytes, displayed as 32 lowercase hex chars (same style as TraceId).
pub type NodeId = TraceId;

/// Operation kind taxonomy (ADR-0017).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    Data,
    Train,
    Comms,
    Governance,
    System,
}

/// Execution trust classification (ADR-0017 / ADR-0018).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTrust {
    #[default]
    Core,
    SandboxedExtension,
    UnsafeExtension,
}

/// A canonical JSON-like value that is stable for hashing and deterministic for serialization.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum CanonValue {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Str(String),
    Array(Vec<CanonValue>),
    Object(BTreeMap<String, CanonValue>),
}

/// Canonical parameters map (stable ordering via BTreeMap).
pub type CanonParams = BTreeMap<String, CanonValue>;

/// Reference to an asset (dataset or intermediate).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetRefV1 {
    /// Stable asset key (e.g., `dataset://<namespace>/<name>`).
    pub asset_key: String,
    /// Optional fingerprint for a specific asset instance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
}

/// A run graph edge (optional; inputs/outputs can imply edges).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EdgeV1 {
    pub from_node_id: NodeId,
    pub to_node_id: NodeId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset_key: Option<String>,
}

/// A single node in the executable run graph (ADR-0017).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeV1 {
    /// Human-stable key (e.g., `prep/clean_users`). This is the stable authoring handle.
    pub node_key: String,
    /// Stable addressing id derived from `node_key`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<NodeId>,

    pub op_kind: OpKind,
    /// Operation type within the kind (e.g., `ingest`, `validate`, `train`, `aggregate`).
    pub op_type: String,

    #[serde(default)]
    pub inputs: Vec<AssetRefV1>,
    #[serde(default)]
    pub outputs: Vec<AssetRefV1>,

    /// Canonical, deterministic params (BTreeMap for stable ordering).
    #[serde(default)]
    pub params: CanonParams,

    /// Optional explicit op code reference (crate/version/op version).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_ref: Option<String>,

    /// Derived flag: whether this node introduces an unsafe surface (untrusted input or unsafe extension).
    #[serde(default)]
    pub unsafe_surface: bool,

    /// Execution trust classification (ADR-0018).
    #[serde(default)]
    pub execution_trust: ExecutionTrust,

    /// Deterministic hash of the operation definition (canonical binary encoding).
    ///
    /// Encoding: lowercase hex of SHA-256 (64 chars).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_def_hash: Option<String>,
}

/// The executable graph file (`graph.json`) schema v1.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphV1 {
    pub schema_version: u32,
    /// Optional stable graph id (may be equal to `run_id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<String>,
    #[serde(default)]
    pub nodes: Vec<NodeV1>,
    #[serde(default)]
    pub edges: Vec<EdgeV1>,
}

impl Default for GraphV1 {
    fn default() -> Self {
        Self {
            schema_version: GRAPH_SCHEMA_V1,
            graph_id: None,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

/// Canonical struct used for node identity hashing.
///
/// IMPORTANT: This excludes runtime-only fields by design.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct NodeDefCanonicalV1<'a> {
    schema_version: u32,
    op_kind: OpKind,
    op_type: &'a str,
    code_ref: &'a str,
    inputs: &'a [AssetRefV1],
    outputs: &'a [AssetRefV1],
    params: &'a CanonParams,
}

/// Compute stable `node_id` from a human-stable `node_key`.
pub fn node_id_from_key(node_key: &str) -> NodeId {
    let digest = Sha256::digest(node_key.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    NodeId::from_bytes(bytes)
}

/// Compute `node_def_hash` for a node (canonical binary encoding).
pub fn node_def_hash_v1(node: &NodeV1) -> [u8; 32] {
    let code_ref = node.code_ref.as_deref().unwrap_or("");
    let canonical = NodeDefCanonicalV1 {
        schema_version: GRAPH_SCHEMA_V1,
        op_kind: node.op_kind,
        op_type: &node.op_type,
        code_ref,
        inputs: &node.inputs,
        outputs: &node.outputs,
        params: &node.params,
    };

    // Postcard provides a deterministic binary encoding when the input types are deterministic
    // (notably: BTreeMap for maps).
    let bytes = match postcard::to_allocvec(&canonical) {
        Ok(bytes) => bytes,
        Err(_) => {
            let mut hasher = Sha256::new();
            hasher.update(b"swarmtorch.node_def_hash_v1.postcard_error");
            hasher.update(node.op_type.as_bytes());
            let digest = hasher.finalize();
            let mut out = [0u8; 32];
            out.copy_from_slice(&digest[..]);
            return out;
        }
    };
    let digest = Sha256::digest(&bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..]);
    out
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Normalize a node for persistence:
/// - fills `node_id` from `node_key`
/// - computes `node_def_hash`
pub fn normalize_node_v1(mut node: NodeV1) -> NodeV1 {
    node.node_id = Some(node_id_from_key(&node.node_key));
    let digest = node_def_hash_v1(&node);
    node.node_def_hash = Some(hex_lower(&digest));
    node
}

impl GraphV1 {
    /// Normalize all nodes (fill derived fields).
    pub fn normalize(mut self) -> Self {
        self.schema_version = GRAPH_SCHEMA_V1;
        self.nodes = self.nodes.into_iter().map(normalize_node_v1).collect();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_is_stable_for_key() {
        let a = node_id_from_key("prep/clean_users").to_string();
        let b = node_id_from_key("prep/clean_users").to_string();
        assert_eq!(a, b);
    }

    #[test]
    fn node_def_hash_changes_with_params() {
        let mut node = NodeV1 {
            node_key: "prep/clean_users".to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "validate".to_string(),
            inputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/users".to_string(),
                fingerprint: None,
            }],
            outputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/users_clean".to_string(),
                fingerprint: None,
            }],
            params: CanonParams::new(),
            code_ref: Some("swarm-torch-data@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
        };

        node = normalize_node_v1(node);
        let h1 = node.node_def_hash.clone().unwrap();

        node.params.insert(
            "null_policy".to_string(),
            CanonValue::Str("drop".to_string()),
        );
        node = normalize_node_v1(node);
        let h2 = node.node_def_hash.clone().unwrap();

        assert_ne!(h1, h2);
    }
}
