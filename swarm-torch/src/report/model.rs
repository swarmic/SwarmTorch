use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use swarm_torch_core::dataops::{
    DatasetLineageV1, DatasetRegistryV1, MaterializationRecordV2, TrustClass, UnsafeReasonV0,
};
use swarm_torch_core::observe::{EventRecord, MetricRecord, SpanRecord};
use swarm_torch_core::run_graph::{ExecutionTrust, GraphV1, NodeV1};

/// Report data loaded from a run artifact bundle.
#[derive(Debug, serde::Serialize)]
pub struct Report {
    #[serde(serialize_with = "serialize_path")]
    pub run_dir: PathBuf,
    pub graph: GraphV1,
    pub registry: DatasetRegistryV1,
    pub lineage: DatasetLineageV1,
    pub materializations: Vec<MaterializationRecordV2>,
    pub spans: Vec<SpanRecord>,
    pub events: Vec<EventRecord>,
    pub metrics: Vec<MetricRecord>,
}

fn serialize_path<S: serde::Serializer>(path: &Path, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&path.display().to_string())
}

/// Derive whether a node should be marked unsafe.
///
/// A node is unsafe if:
/// - `execution_trust != Core`, OR
/// - any input asset_key is `Untrusted` in the registry, OR
/// - any input asset_key is **missing** from the registry (fail closed).
pub fn is_node_unsafe(node: &NodeV1, registry: &DatasetRegistryV1) -> bool {
    let trust_index = build_registry_trust_index(registry);
    is_node_unsafe_with_index(node, &trust_index)
}

pub(crate) fn build_registry_trust_index(
    registry: &DatasetRegistryV1,
) -> BTreeMap<String, TrustClass> {
    registry
        .datasets
        .iter()
        .map(|entry| (entry.asset_key.clone(), entry.trust))
        .collect()
}

pub(crate) fn is_node_unsafe_with_index(
    node: &NodeV1,
    trust_index: &BTreeMap<String, TrustClass>,
) -> bool {
    if node.execution_trust != ExecutionTrust::Core {
        return true;
    }
    for input in &node.inputs {
        match trust_index.get(&input.asset_key).copied() {
            Some(TrustClass::Untrusted) => return true,
            None => return true, // missing input -> fail closed
            _ => {}
        }
    }
    false
}

pub(crate) fn unsafe_reason_label(reason: UnsafeReasonV0) -> &'static str {
    match reason {
        UnsafeReasonV0::UntrustedInput => "untrusted_input",
        UnsafeReasonV0::UnsafeExtension => "unsafe_extension",
        UnsafeReasonV0::MissingProvenance => "missing_provenance",
    }
}

pub(crate) fn format_unsafe_reasons(reasons: &[UnsafeReasonV0]) -> String {
    if reasons.is_empty() {
        return "none".to_string();
    }
    reasons
        .iter()
        .map(|reason| unsafe_reason_label(*reason))
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn format_transform_names(
    transforms: &[swarm_torch_core::dataops::TransformAuditV0],
) -> String {
    if transforms.is_empty() {
        return "none".to_string();
    }
    transforms
        .iter()
        .map(|t| t.transform_name.as_str())
        .collect::<Vec<_>>()
        .join(",")
}
