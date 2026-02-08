//! Execution seams (ADR-0018): policy + runner boundaries.
//!
//! This module intentionally does **not** implement a scheduler or DAG engine.
//! It defines call boundaries that let us:
//! - enforce policy decisions before executing nodes
//! - keep execution auditable/testable
//! - swap native vs sandboxed runners later without changing `graph.json` semantics

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
