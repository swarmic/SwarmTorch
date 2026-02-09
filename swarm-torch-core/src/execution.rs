//! Execution seams (ADR-0018): policy + runner boundaries.
//!
//! This module intentionally does **not** implement a scheduler or DAG engine.
//! It defines call boundaries that let us:
//! - enforce policy decisions before executing nodes
//! - keep execution auditable/testable
//! - swap native vs sandboxed runners later without changing `graph.json` semantics

#[cfg(feature = "alloc")]
use alloc::format;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::dataops::DatasetRegistryV1;
use crate::observe::RunEventEmitter;
use crate::run_graph::NodeV1;

/// Policy decision for whether a node may execute under the current profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    Allowed,
    Denied { reason: String },
}

/// Execution policy boundary (ADR-0018).
pub trait ExecutionPolicy: Send + Sync {
    fn allow(&self, node: &NodeV1, registry: &DatasetRegistryV1) -> PolicyDecision;
}

/// A runtime-resolved asset instance.
///
/// This is metadata-only: the actual payload is always a pointer + hash, not embedded bytes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetInstanceV1 {
    pub asset_key: String,
    pub fingerprint_v0: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

/// Node runner boundary (ADR-0018).
///
/// NOTE: `RunEventEmitter` keeps this sink-agnostic. A std artifact sink can persist NDJSON,
/// and a future exporter can map to OTLP.
pub trait OpRunner: Send + Sync {
    type Error;

    fn run<E: RunEventEmitter<Error = Self::Error>>(
        &self,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        emitter: &E,
    ) -> core::result::Result<Vec<AssetInstanceV1>, Self::Error>;
}

// ---------------------------------------------------------------------------
// ExecutionPolicy implementations
// ---------------------------------------------------------------------------

use crate::run_graph::ExecutionTrust;

/// Default policy: Core trust only; untrusted inputs allowed but DataOps marks unsafe_surface.
///
/// Use with `DataOpsSession::materialize_node_outputs()` which propagates trust correctly.
pub struct CoreOnlyPolicy;

impl ExecutionPolicy for CoreOnlyPolicy {
    fn allow(&self, node: &NodeV1, _registry: &DatasetRegistryV1) -> PolicyDecision {
        match node.execution_trust {
            ExecutionTrust::Core => PolicyDecision::Allowed,
            _ => PolicyDecision::Denied {
                reason: format!(
                    "node {} requires Core trust, has {:?}",
                    node.node_key, node.execution_trust
                ),
            },
        }
    }
}

/// Permissive policy for tests and development (allows all nodes).
pub struct PermissivePolicy;

impl ExecutionPolicy for PermissivePolicy {
    fn allow(&self, _node: &NodeV1, _registry: &DatasetRegistryV1) -> PolicyDecision {
        PolicyDecision::Allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_graph::{CanonParams, NodeV1, OpKind};

    fn test_node(trust: ExecutionTrust) -> NodeV1 {
        NodeV1 {
            node_key: "test/node".to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "test".to_string(),
            inputs: vec![],
            outputs: vec![],
            params: CanonParams::new(),
            code_ref: Some("test@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: trust,
            node_def_hash: None,
        }
    }

    #[test]
    fn policy_allows_core_trust() {
        let policy = CoreOnlyPolicy;
        let registry = DatasetRegistryV1::default();
        let node = test_node(ExecutionTrust::Core);
        assert_eq!(policy.allow(&node, &registry), PolicyDecision::Allowed);
    }

    #[test]
    fn policy_denies_sandboxed_extension() {
        let policy = CoreOnlyPolicy;
        let registry = DatasetRegistryV1::default();
        let node = test_node(ExecutionTrust::SandboxedExtension);
        match policy.allow(&node, &registry) {
            PolicyDecision::Denied { reason } => {
                assert!(reason.contains("Core trust"));
            }
            _ => panic!("expected Denied"),
        }
    }

    #[test]
    fn policy_denies_unsafe_extension() {
        let policy = CoreOnlyPolicy;
        let registry = DatasetRegistryV1::default();
        let node = test_node(ExecutionTrust::UnsafeExtension);
        match policy.allow(&node, &registry) {
            PolicyDecision::Denied { reason } => {
                assert!(reason.contains("Core trust"));
            }
            _ => panic!("expected Denied"),
        }
    }

    #[test]
    fn permissive_allows_all() {
        let policy = PermissivePolicy;
        let registry = DatasetRegistryV1::default();
        assert_eq!(
            policy.allow(&test_node(ExecutionTrust::Core), &registry),
            PolicyDecision::Allowed
        );
        assert_eq!(
            policy.allow(&test_node(ExecutionTrust::SandboxedExtension), &registry),
            PolicyDecision::Allowed
        );
        assert_eq!(
            policy.allow(&test_node(ExecutionTrust::UnsafeExtension), &registry),
            PolicyDecision::Allowed
        );
    }
}
