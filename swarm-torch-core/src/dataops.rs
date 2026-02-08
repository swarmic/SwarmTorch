//! DataOps metadata: dataset registry, lineage, materializations, fingerprints.
//!
//! This module is intentionally "metadata-first":
//! - fingerprints must be computable without raw rows
//! - lineage/materialization records power observability + reproducibility
//! - execution can be added later (ADR-0018) without changing these schemas
//!
//! Fingerprint v0 (pragmatic + stable):
//! - `source_fingerprint_v0` = sha256(postcard(normalized source descriptor))
//! - `schema_hash_v0` = sha256(postcard(normalized schema descriptor))
//! - `recipe_hash_v0` = sha256(postcard({ node_def_hash, upstream_fingerprints }))
//! - `dataset_fingerprint_v0` = sha256(postcard({ source_fingerprint, schema_hash, recipe_hash }))

#[cfg(feature = "alloc")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::format;
#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sha2::{Digest, Sha256};

use crate::run_graph::{node_def_hash_v1, NodeId, NodeV1, OpKind};

pub const DATAOPS_SCHEMA_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustClass {
    Trusted,
    Untrusted,
}

impl Default for TrustClass {
    fn default() -> Self {
        Self::Trusted
    }
}

/// Authentication mode marker (DO NOT put secrets here).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthModeMarker {
    None,
    BearerToken,
    Basic,
    Mtls,
    Custom(String),
}

impl Default for AuthModeMarker {
    fn default() -> Self {
        Self::None
    }
}

/// Dataset source descriptor used for source fingerprinting.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceDescriptorV0 {
    pub uri: String,
    pub content_type: String,
    #[serde(default)]
    pub auth_mode: AuthModeMarker,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub etag_or_version: Option<String>,
}

/// Schema descriptor used for schema hashing.
///
/// `canonical` SHOULD be a canonical representation for stable hashing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SchemaDescriptorV0 {
    /// e.g. `arrow-json`, `json-schema`, `parquet-thrift`
    pub format: String,
    /// canonical, stable representation (no raw rows; schema only)
    pub canonical: String,
}

/// One dataset/asset entry in `datasets/registry.json` (schema v1).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DatasetEntryV1 {
    pub asset_key: String,

    /// Dataset fingerprint v0 (lowercase hex sha256).
    pub fingerprint_v0: String,
    pub source_fingerprint_v0: String,
    pub schema_hash_v0: String,
    pub recipe_hash_v0: String,

    #[serde(default)]
    pub trust: TrustClass,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceDescriptorV0>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<SchemaDescriptorV0>,

    /// Optional metadata tags.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub license_flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pii_tags: Vec<String>,
}

/// `datasets/registry.json` schema v1.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DatasetRegistryV1 {
    pub schema_version: u32,
    #[serde(default)]
    pub datasets: Vec<DatasetEntryV1>,
}

impl Default for DatasetRegistryV1 {
    fn default() -> Self {
        Self {
            schema_version: DATAOPS_SCHEMA_V1,
            datasets: Vec::new(),
        }
    }
}

/// A lineage edge: input fingerprint -> output fingerprint via a node.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LineageEdgeV1 {
    pub input_fingerprint_v0: String,
    pub output_fingerprint_v0: String,
    pub node_id: NodeId,
    pub op_kind: OpKind,
}

/// `datasets/lineage.json` schema v1.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DatasetLineageV1 {
    pub schema_version: u32,
    #[serde(default)]
    pub edges: Vec<LineageEdgeV1>,
}

impl Default for DatasetLineageV1 {
    fn default() -> Self {
        Self {
            schema_version: DATAOPS_SCHEMA_V1,
            edges: Vec::new(),
        }
    }
}

/// One materialization record per node output (NDJSON line schema v1).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterializationRecordV1 {
    pub schema_version: u32,
    pub ts_unix_nanos: u64,

    pub asset_key: String,
    pub fingerprint_v0: String,

    pub node_id: NodeId,
    pub node_def_hash: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_flags: Option<Vec<String>>,

    #[serde(default)]
    pub unsafe_surface: bool,
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

fn sha256_postcard<T: serde::Serialize>(value: &T) -> [u8; 32] {
    let bytes = postcard::to_allocvec(value).expect("postcard serialization must succeed");
    let digest = Sha256::digest(&bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..]);
    out
}

fn normalize_lower(s: &str) -> String {
    // ASCII-lowercase is sufficient for protocol-ish strings (content types, format tags).
    s.trim().to_ascii_lowercase()
}

fn normalize_trim(s: &str) -> String {
    s.trim().to_string()
}

fn auth_mode_marker_str(m: &AuthModeMarker) -> String {
    match m {
        AuthModeMarker::None => "none".to_string(),
        AuthModeMarker::BearerToken => "bearer_token".to_string(),
        AuthModeMarker::Basic => "basic".to_string(),
        AuthModeMarker::Mtls => "mtls".to_string(),
        AuthModeMarker::Custom(v) => format!("custom:{v}"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct SourceFingerprintCanonicalV0 {
    uri: String,
    content_type: String,
    auth_mode: String,
    etag_or_version: Option<String>,
}

/// Compute `source_fingerprint` (v0) from a normalized source descriptor.
pub fn source_fingerprint_v0(source: &SourceDescriptorV0) -> [u8; 32] {
    let canonical = SourceFingerprintCanonicalV0 {
        uri: normalize_trim(&source.uri),
        content_type: normalize_lower(&source.content_type),
        auth_mode: auth_mode_marker_str(&source.auth_mode),
        etag_or_version: source.etag_or_version.as_ref().map(|v| normalize_trim(v)),
    };
    sha256_postcard(&canonical)
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct SchemaHashCanonicalV0 {
    format: String,
    canonical: String,
}

/// Compute `schema_hash` (v0) from a normalized schema descriptor.
pub fn schema_hash_v0(schema: &SchemaDescriptorV0) -> [u8; 32] {
    let canonical = SchemaHashCanonicalV0 {
        format: normalize_lower(&schema.format),
        canonical: normalize_trim(&schema.canonical),
    };
    sha256_postcard(&canonical)
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct RecipeHashCanonicalV0 {
    node_def_hash: [u8; 32],
    upstream_fingerprints: Vec<[u8; 32]>,
}

/// Compute `recipe_hash` (v0) for a transform definition.
///
/// This is stable without raw rows: it depends on the node definition hash and upstream fingerprints.
pub fn recipe_hash_v0(node: &NodeV1, upstream_fingerprints: &[[u8; 32]]) -> [u8; 32] {
    let node_def_hash = node_def_hash_v1(node);
    let canonical = RecipeHashCanonicalV0 {
        node_def_hash,
        upstream_fingerprints: upstream_fingerprints.to_vec(),
    };
    sha256_postcard(&canonical)
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct DatasetFingerprintCanonicalV0 {
    source_fingerprint: [u8; 32],
    schema_hash: [u8; 32],
    recipe_hash: [u8; 32],
}

/// Compute dataset fingerprint v0.
pub fn dataset_fingerprint_v0(
    source_fingerprint: [u8; 32],
    schema_hash: [u8; 32],
    recipe_hash: [u8; 32],
) -> [u8; 32] {
    let canonical = DatasetFingerprintCanonicalV0 {
        source_fingerprint,
        schema_hash,
        recipe_hash,
    };
    sha256_postcard(&canonical)
}

/// Convenience: build a registry entry (v1) from the provided descriptors.
pub fn dataset_entry_v1(
    asset_key: impl Into<String>,
    trust: TrustClass,
    source: Option<SourceDescriptorV0>,
    schema: Option<SchemaDescriptorV0>,
    recipe_hash: [u8; 32],
) -> DatasetEntryV1 {
    let asset_key = asset_key.into();

    let source_fp = source
        .as_ref()
        .map(source_fingerprint_v0)
        .unwrap_or_else(|| sha256_postcard(&"no_source"));

    let schema_hash = schema
        .as_ref()
        .map(schema_hash_v0)
        .unwrap_or_else(|| sha256_postcard(&"no_schema"));

    let dataset_fp = dataset_fingerprint_v0(source_fp, schema_hash, recipe_hash);

    DatasetEntryV1 {
        asset_key,
        fingerprint_v0: hex_lower(&dataset_fp),
        source_fingerprint_v0: hex_lower(&source_fp),
        schema_hash_v0: hex_lower(&schema_hash),
        recipe_hash_v0: hex_lower(&recipe_hash),
        trust,
        source,
        schema,
        license_flags: Vec::new(),
        pii_tags: Vec::new(),
    }
}

/// Minimal canonical params helper (BTreeMap ordering is deterministic).
pub fn canon_params_from_pairs(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v.to_string());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_graph::{AssetRefV1, CanonParams, ExecutionTrust, NodeV1};

    #[test]
    fn dataset_fingerprint_is_deterministic() {
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/path".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::BearerToken,
            etag_or_version: Some("etag123".to_string()),
        };
        let schema = SchemaDescriptorV0 {
            format: "arrow-json".to_string(),
            canonical: "{\"fields\":[{\"name\":\"x\",\"type\":\"i64\"}]}".to_string(),
        };

        let node = NodeV1 {
            node_key: "prep/clean".to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "validate".to_string(),
            inputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/raw".to_string(),
                fingerprint: None,
            }],
            outputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/clean".to_string(),
                fingerprint: None,
            }],
            params: CanonParams::new(),
            code_ref: Some("swarm-torch-data@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
        };

        let upstream = [[7u8; 32]];
        let recipe = recipe_hash_v0(&node, &upstream);

        let a = dataset_entry_v1(
            "dataset://ns/clean",
            TrustClass::Trusted,
            Some(source.clone()),
            Some(schema.clone()),
            recipe,
        );
        let b = dataset_entry_v1(
            "dataset://ns/clean",
            TrustClass::Trusted,
            Some(source),
            Some(schema),
            recipe,
        );
        assert_eq!(a.fingerprint_v0, b.fingerprint_v0);
        assert_eq!(a.schema_hash_v0, b.schema_hash_v0);
        assert_eq!(a.source_fingerprint_v0, b.source_fingerprint_v0);
    }
}
