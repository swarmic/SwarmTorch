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

/// Target portability profile for node scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferredProfile {
    EmbeddedMin,
    EmbeddedAlloc,
    EdgeStd,
}

/// Device role affinity hint (informational; planner may override).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceAffinity {
    Coordinator,
    AnyParticipant,
}

/// Optional scheduling hint for run-graph planning.
///
/// This metadata is intentionally excluded from `node_def_hash` and cache identity.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionHint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_profile: Option<PreferredProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_affinity: Option<DeviceAffinity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_budget_bytes: Option<u64>,
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

    /// Optional scheduling hint — excluded from `node_def_hash` and cache identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_hint: Option<ExecutionHint>,
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
    // execution_hint is intentionally excluded: it is planner metadata, not op identity (F1).
}

/// Derive a stable [`NodeId`] from a human-readable `node_key`.
///
/// # Collision characteristics (M-08)
///
/// `NodeId` is 128-bit (16 bytes) derived from `SHA-256(node_key)` truncated to
/// the first 16 bytes. The birthday-bound collision probability for `n` nodes is
/// approximately `n² / 2¹²⁹`. For 10,000 nodes this is ~2⁻¹⁰³ — negligible
/// for all practical graph sizes. If collision avoidance is ever required for
/// cross-graph identity, consider using the full 32-byte SHA-256 digest.
pub fn node_id_from_key(node_key: &str) -> NodeId {
    let digest = Sha256::digest(node_key.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    NodeId::from_bytes(bytes)
}

/// Compute `node_def_hash` for a node (canonical binary encoding).
///
/// Excludes runtime/planner metadata such as `execution_hint`.
pub fn node_def_hash_v1(node: &NodeV1) -> Result<[u8; 32], postcard::Error> {
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
    let bytes = postcard::to_allocvec(&canonical)?;
    let digest = Sha256::digest(&bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..]);
    Ok(out)
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
pub fn normalize_node_v1(mut node: NodeV1) -> Result<NodeV1, postcard::Error> {
    node.node_id = Some(node_id_from_key(&node.node_key));
    let digest = node_def_hash_v1(&node)?;
    node.node_def_hash = Some(hex_lower(&digest));
    Ok(node)
}

// ── NodeV1 write-path validation (M-09) ──────────────────────────

/// Maximum length for `NodeV1.node_key`.
pub const MAX_NODE_KEY_LEN: usize = 256;
/// Maximum length for `NodeV1.op_type`.
pub const MAX_OP_TYPE_LEN: usize = 128;
/// Maximum number of inputs per node.
pub const MAX_NODE_INPUTS: usize = 256;
/// Maximum number of outputs per node.
pub const MAX_NODE_OUTPUTS: usize = 256;
/// Maximum number of entries in `NodeV1.params`.
pub const MAX_PARAMS_COUNT: usize = 128;

/// Validation error for `NodeV1` field bounds.
///
/// These checks are applied on the **write path** only (graph construction /
/// artifact persistence). The read path (`load_report` → `normalize`) remains
/// tolerant to avoid breaking historical bundles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeValidationError {
    /// `node_key` exceeds `MAX_NODE_KEY_LEN` chars.
    NodeKeyTooLong { len: usize, max: usize },
    /// `op_type` exceeds `MAX_OP_TYPE_LEN` chars.
    OpTypeTooLong { len: usize, max: usize },
    /// Number of inputs exceeds `MAX_NODE_INPUTS`.
    TooManyInputs { count: usize, max: usize },
    /// Number of outputs exceeds `MAX_NODE_OUTPUTS`.
    TooManyOutputs { count: usize, max: usize },
    /// Number of params exceeds `MAX_PARAMS_COUNT`.
    TooManyParams { count: usize, max: usize },
}

impl core::fmt::Display for NodeValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NodeKeyTooLong { len, max } => {
                write!(f, "node_key length {len} exceeds maximum {max}")
            }
            Self::OpTypeTooLong { len, max } => {
                write!(f, "op_type length {len} exceeds maximum {max}")
            }
            Self::TooManyInputs { count, max } => {
                write!(f, "input count {count} exceeds maximum {max}")
            }
            Self::TooManyOutputs { count, max } => {
                write!(f, "output count {count} exceeds maximum {max}")
            }
            Self::TooManyParams { count, max } => {
                write!(f, "params count {count} exceeds maximum {max}")
            }
        }
    }
}

/// Validate `NodeV1` field bounds (write-path only).
///
/// Returns `Ok(())` if all fields are within acceptable limits.
///
/// This function does NOT mutate the node. Call it before `normalize_node_v1`
/// on the write path (graph construction / artifact persistence).
pub fn validate_node_v1(node: &NodeV1) -> Result<(), NodeValidationError> {
    if node.node_key.len() > MAX_NODE_KEY_LEN {
        return Err(NodeValidationError::NodeKeyTooLong {
            len: node.node_key.len(),
            max: MAX_NODE_KEY_LEN,
        });
    }
    if node.op_type.len() > MAX_OP_TYPE_LEN {
        return Err(NodeValidationError::OpTypeTooLong {
            len: node.op_type.len(),
            max: MAX_OP_TYPE_LEN,
        });
    }
    if node.inputs.len() > MAX_NODE_INPUTS {
        return Err(NodeValidationError::TooManyInputs {
            count: node.inputs.len(),
            max: MAX_NODE_INPUTS,
        });
    }
    if node.outputs.len() > MAX_NODE_OUTPUTS {
        return Err(NodeValidationError::TooManyOutputs {
            count: node.outputs.len(),
            max: MAX_NODE_OUTPUTS,
        });
    }
    if node.params.len() > MAX_PARAMS_COUNT {
        return Err(NodeValidationError::TooManyParams {
            count: node.params.len(),
            max: MAX_PARAMS_COUNT,
        });
    }
    Ok(())
}

impl GraphV1 {
    /// Normalize all nodes (fill derived fields).
    pub fn normalize(mut self) -> Result<Self, postcard::Error> {
        self.schema_version = GRAPH_SCHEMA_V1;
        self.nodes = self
            .nodes
            .into_iter()
            .map(normalize_node_v1)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
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
            execution_hint: None,
        };

        node = normalize_node_v1(node).unwrap();
        let h1 = node.node_def_hash.clone().unwrap();

        node.params.insert(
            "null_policy".to_string(),
            CanonValue::Str("drop".to_string()),
        );
        node = normalize_node_v1(node).unwrap();
        let h2 = node.node_def_hash.clone().unwrap();

        assert_ne!(h1, h2);
    }

    // ── M-09 validation tests ──

    fn make_valid_node() -> NodeV1 {
        NodeV1 {
            node_key: "prep/clean".to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "validate".to_string(),
            inputs: vec![],
            outputs: vec![],
            params: CanonParams::new(),
            code_ref: None,
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
            execution_hint: None,
        }
    }

    #[test]
    fn valid_node_passes_validation() {
        assert!(validate_node_v1(&make_valid_node()).is_ok());
    }

    #[test]
    fn node_key_too_long_rejected() {
        let mut node = make_valid_node();
        node.node_key = "x".repeat(MAX_NODE_KEY_LEN + 1);
        assert_eq!(
            validate_node_v1(&node),
            Err(NodeValidationError::NodeKeyTooLong {
                len: MAX_NODE_KEY_LEN + 1,
                max: MAX_NODE_KEY_LEN,
            })
        );
    }

    #[test]
    fn op_type_too_long_rejected() {
        let mut node = make_valid_node();
        node.op_type = "x".repeat(MAX_OP_TYPE_LEN + 1);
        assert_eq!(
            validate_node_v1(&node),
            Err(NodeValidationError::OpTypeTooLong {
                len: MAX_OP_TYPE_LEN + 1,
                max: MAX_OP_TYPE_LEN,
            })
        );
    }

    #[test]
    fn too_many_inputs_rejected() {
        let mut node = make_valid_node();
        node.inputs = (0..MAX_NODE_INPUTS + 1)
            .map(|i| AssetRefV1 {
                asset_key: format!("dataset://ns/{i}"),
                fingerprint: None,
            })
            .collect();
        assert_eq!(
            validate_node_v1(&node),
            Err(NodeValidationError::TooManyInputs {
                count: MAX_NODE_INPUTS + 1,
                max: MAX_NODE_INPUTS,
            })
        );
    }

    #[test]
    fn execution_hint_excluded_from_node_def_hash() {
        let mut a = make_valid_node();
        let mut b = a.clone();
        b.execution_hint = Some(ExecutionHint {
            preferred_profile: Some(PreferredProfile::EdgeStd),
            device_affinity: Some(DeviceAffinity::Coordinator),
            memory_budget_bytes: Some(65_536),
        });

        let ha = node_def_hash_v1(&a).unwrap();
        let hb = node_def_hash_v1(&b).unwrap();
        assert_eq!(ha, hb, "execution_hint must not affect node_def_hash");

        a = normalize_node_v1(a).unwrap();
        b = normalize_node_v1(b).unwrap();
        assert_eq!(a.node_def_hash, b.node_def_hash);
    }

    #[test]
    fn node_v1_roundtrips_with_full_execution_hint() {
        let mut node = make_valid_node();
        node.execution_hint = Some(ExecutionHint {
            preferred_profile: Some(PreferredProfile::EmbeddedAlloc),
            device_affinity: Some(DeviceAffinity::AnyParticipant),
            memory_budget_bytes: Some(32_768),
        });

        let json = serde_json::to_string(&node).unwrap();
        let decoded: NodeV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.execution_hint, node.execution_hint);
    }

    #[test]
    fn node_v1_roundtrips_with_no_execution_hint() {
        let node = make_valid_node();
        let json = serde_json::to_string(&node).unwrap();
        assert!(
            !json.contains("execution_hint"),
            "execution_hint should be omitted when None"
        );
        let decoded: NodeV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.execution_hint, None);
    }

    #[test]
    fn graph_json_backward_compatible_without_hint_field() {
        let json = r#"{
            "schema_version": 1,
            "nodes": [{
                "node_key": "prep/clean",
                "op_kind": "data",
                "op_type": "validate",
                "inputs": [],
                "outputs": [],
                "params": {},
                "unsafe_surface": false,
                "execution_trust": "core"
            }],
            "edges": []
        }"#;
        let graph: GraphV1 = serde_json::from_str(json).unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.nodes[0].execution_hint.is_none());
    }
}
