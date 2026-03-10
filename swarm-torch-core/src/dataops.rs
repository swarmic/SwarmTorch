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
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use sha2::{Digest, Sha256};

use crate::run_graph::{node_def_hash_v1, NodeId, NodeV1, OpKind};

pub const DATAOPS_SCHEMA_V1: u32 = 1;
pub const MATERIALIZATION_SCHEMA_V2: u32 = 2;
pub const MAX_SOURCE_URI_LEN: usize = 2048;
pub const MAX_ETAG_OR_VERSION_LEN: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrustClass {
    #[default]
    Trusted,
    Untrusted,
}

/// Authentication mode marker (DO NOT put secrets here).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AuthModeMarker {
    #[default]
    None,
    BearerToken,
    Basic,
    Mtls,
    Custom(String),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDecisionV0 {
    Hit,
    Miss,
    Bypass,
    Unknown,
}

impl From<bool> for CacheDecisionV0 {
    fn from(value: bool) -> Self {
        if value {
            Self::Hit
        } else {
            Self::Miss
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterializationStatusV0 {
    Ok,
    Error,
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnsafeReasonV0 {
    UntrustedInput,
    UnsafeExtension,
    MissingProvenance,
}

/// Audit metadata for update-transform application.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TransformAuditV0 {
    /// Human-readable transform name.
    pub transform_name: String,
    /// Whether this transform is trusted as a core transform.
    pub core_trusted: bool,
    /// Round in which the transform was applied.
    pub round_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceDescriptorError {
    UriTooLong { len: usize, max: usize },
    EtagOrVersionTooLong { len: usize, max: usize },
}

impl core::fmt::Display for SourceDescriptorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UriTooLong { len, max } => {
                write!(f, "source descriptor uri too long: {} > {}", len, max)
            }
            Self::EtagOrVersionTooLong { len, max } => {
                write!(
                    f,
                    "source descriptor etag_or_version too long: {} > {}",
                    len, max
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SourceDescriptorError {}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QualitySummaryV0 {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub null_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_count_delta: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_changed: Option<bool>,
}

/// One materialization record per node output (NDJSON line schema v2).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MaterializationRecordV2 {
    pub schema_version: u32, // = 2
    pub record_seq: u64,     // monotonic per run
    pub ts_unix_nanos: u64,

    pub asset_key: String,
    pub fingerprint_v0: String,
    pub node_id: NodeId,
    pub node_def_hash: String,
    pub op_type: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_asset_keys: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_fingerprints_v0: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,

    pub cache_decision: CacheDecisionV0,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_key_v0: Option<String>,

    /// Compatibility convenience: derived from `cache_decision`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,

    pub unsafe_surface: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unsafe_reasons: Vec<UnsafeReasonV0>,
    /// Applied update transforms (if any) for this materialization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applied_transforms: Vec<TransformAuditV0>,

    pub status: MaterializationStatusV0,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualitySummaryV0>,
}

/// Compatibility reader for `datasets/materializations.ndjson`.
///
/// Ordering is important: V2 MUST come first so V2 rows are not down-cast to V1.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum MaterializationRecordCompat {
    V2(MaterializationRecordV2),
    V1(MaterializationRecordV1),
}

impl MaterializationRecordCompat {
    /// Normalize compatibility records into the v2 in-memory shape.
    pub fn into_v2(self) -> MaterializationRecordV2 {
        match self {
            Self::V2(v2) => v2,
            Self::V1(v1) => MaterializationRecordV2 {
                schema_version: MATERIALIZATION_SCHEMA_V2,
                record_seq: 0,
                ts_unix_nanos: v1.ts_unix_nanos,
                asset_key: v1.asset_key,
                fingerprint_v0: v1.fingerprint_v0,
                node_id: v1.node_id,
                node_def_hash: v1.node_def_hash,
                op_type: "unknown".to_string(),
                input_asset_keys: Vec::new(),
                input_fingerprints_v0: Vec::new(),
                rows: v1.rows,
                bytes: v1.bytes,
                duration_ms: v1.duration_ms,
                cache_decision: match v1.cache_hit {
                    Some(true) => CacheDecisionV0::Hit,
                    Some(false) => CacheDecisionV0::Miss,
                    None => CacheDecisionV0::Unknown,
                },
                cache_reason: None,
                cache_key_v0: None,
                cache_hit: v1.cache_hit,
                unsafe_surface: true,
                unsafe_reasons: vec![UnsafeReasonV0::MissingProvenance],
                applied_transforms: Vec::new(),
                status: MaterializationStatusV0::Ok,
                error_code: None,
                quality: None,
            },
        }
    }
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

fn sha256_postcard<T: serde::Serialize>(value: &T) -> Result<[u8; 32], postcard::Error> {
    let bytes = postcard::to_allocvec(value)?;
    let digest = Sha256::digest(&bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..]);
    Ok(out)
}

fn normalize_lower(s: &str) -> String {
    // ASCII-lowercase is sufficient for protocol-ish strings (content types, format tags).
    s.trim().to_ascii_lowercase()
}

fn normalize_trim(s: &str) -> String {
    s.trim().to_string()
}

fn redact_uri_userinfo(uri: &str) -> String {
    let Some(scheme_sep) = uri.find("://") else {
        return uri.to_string();
    };

    let authority_start = scheme_sep + 3;
    let after_scheme = &uri[authority_start..];

    let mut authority_end = after_scheme.len();
    for delim in ['/', '?', '#'] {
        if let Some(idx) = after_scheme.find(delim) {
            authority_end = authority_end.min(idx);
        }
    }

    let authority = &after_scheme[..authority_end];
    let Some((_, host_part)) = authority.rsplit_once('@') else {
        return uri.to_string();
    };

    let mut out = String::with_capacity(uri.len());
    out.push_str(&uri[..authority_start]);
    out.push_str("<redacted>@");
    out.push_str(host_part);
    out.push_str(&after_scheme[authority_end..]);
    out
}

fn strip_query_fragment(uri: &str) -> &str {
    let end = uri
        .find('?')
        .unwrap_or(uri.len())
        .min(uri.find('#').unwrap_or(uri.len()));
    &uri[..end]
}

fn normalize_and_redact_source_descriptor(source: &SourceDescriptorV0) -> SourceDescriptorV0 {
    let redacted = redact_uri_userinfo(&normalize_trim(&source.uri));
    SourceDescriptorV0 {
        uri: strip_query_fragment(&redacted).to_string(),
        content_type: normalize_lower(&source.content_type),
        auth_mode: source.auth_mode.clone(),
        etag_or_version: source.etag_or_version.as_ref().map(|v| normalize_trim(v)),
    }
}

/// Sanitize a source descriptor before persistence/hashing.
///
/// Guarantees:
/// - URI userinfo is redacted (`user[:pass]@` -> `<redacted>@`)
/// - normalized whitespace/casing rules match fingerprint canonicalization
/// - oversized URI / etag_or_version values are rejected
pub fn sanitize_source_descriptor_v0(
    source: &SourceDescriptorV0,
) -> core::result::Result<SourceDescriptorV0, SourceDescriptorError> {
    let sanitized = normalize_and_redact_source_descriptor(source);
    check_descriptor_bounds(&sanitized)?;
    Ok(sanitized)
}

/// Shared bounds check for source descriptors.
///
/// Used by both the write-path (`sanitize_source_descriptor_v0`) and
/// the read-path (`validate_source_descriptor_bounds`) to ensure
/// one validation core with no drift.
fn check_descriptor_bounds(
    desc: &SourceDescriptorV0,
) -> core::result::Result<(), SourceDescriptorError> {
    if desc.uri.len() > MAX_SOURCE_URI_LEN {
        return Err(SourceDescriptorError::UriTooLong {
            len: desc.uri.len(),
            max: MAX_SOURCE_URI_LEN,
        });
    }
    if let Some(etag_or_version) = desc.etag_or_version.as_ref() {
        if etag_or_version.len() > MAX_ETAG_OR_VERSION_LEN {
            return Err(SourceDescriptorError::EtagOrVersionTooLong {
                len: etag_or_version.len(),
                max: MAX_ETAG_OR_VERSION_LEN,
            });
        }
    }
    Ok(())
}

/// Validate source descriptor bounds without mutation (read-path check).
///
/// Returns `Ok(())` if the descriptor's URI and etag/version lengths are within
/// acceptable limits. This function does not sanitize or redact — it is intended
/// for use at deserialization boundaries (e.g., `load_report`).
pub fn validate_source_descriptor_bounds(
    desc: &SourceDescriptorV0,
) -> core::result::Result<(), SourceDescriptorError> {
    check_descriptor_bounds(desc)
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
pub fn source_fingerprint_v0(source: &SourceDescriptorV0) -> Result<[u8; 32], postcard::Error> {
    let source = normalize_and_redact_source_descriptor(source);
    let canonical = SourceFingerprintCanonicalV0 {
        uri: source.uri,
        content_type: normalize_lower(&source.content_type),
        auth_mode: auth_mode_marker_str(&source.auth_mode),
        etag_or_version: source.etag_or_version,
    };
    sha256_postcard(&canonical)
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct SchemaHashCanonicalV0 {
    format: String,
    canonical: String,
}

/// Compute `schema_hash` (v0) from a normalized schema descriptor.
pub fn schema_hash_v0(schema: &SchemaDescriptorV0) -> Result<[u8; 32], postcard::Error> {
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
pub fn recipe_hash_v0(
    node: &NodeV1,
    upstream_fingerprints: &[[u8; 32]],
) -> Result<[u8; 32], postcard::Error> {
    let node_def_hash = node_def_hash_v1(node)?;
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
) -> Result<[u8; 32], postcard::Error> {
    let canonical = DatasetFingerprintCanonicalV0 {
        source_fingerprint,
        schema_hash,
        recipe_hash,
    };
    sha256_postcard(&canonical)
}

// ---------------------------------------------------------------------------
// Canonical placeholder helpers (single source of truth for fingerprint rules)
// ---------------------------------------------------------------------------

/// Placeholder schema hash when schema descriptor is absent.
///
/// **ADR-0017:** `sha256(postcard("no_schema_v0"))`
pub fn no_schema_hash_v0() -> Result<[u8; 32], postcard::Error> {
    sha256_postcard(&"no_schema_v0")
}

/// Placeholder source fingerprint for derived (non-source) datasets.
///
/// Salted with `asset_key` to differentiate multi-output nodes.
///
/// **ADR-0017:** `sha256(postcard("derived_v0:{asset_key}"))`
pub fn derived_source_fingerprint_v0(asset_key: &str) -> Result<[u8; 32], postcard::Error> {
    sha256_postcard(&format!("derived_v0:{}", asset_key))
}

/// Convenience: build a registry entry (v1) from the provided descriptors.
///
/// **Warning:** For derived outputs (source = None), use `derived_dataset_entry_v1()` instead
/// to ensure proper asset_key salting in source_fingerprint.
pub fn dataset_entry_v1(
    asset_key: impl Into<String>,
    trust: TrustClass,
    source: Option<SourceDescriptorV0>,
    schema: Option<SchemaDescriptorV0>,
    recipe_hash: [u8; 32],
) -> Result<DatasetEntryV1, postcard::Error> {
    let asset_key = asset_key.into();

    // Use canonical helper for missing schema
    let schema_fp = match schema.as_ref() {
        Some(s) => schema_hash_v0(s)?,
        None => no_schema_hash_v0()?,
    };

    // For source: if None, this function uses a non-salted placeholder
    // which is ONLY correct for root sources without upstream.
    // For derived outputs, callers SHOULD use derived_dataset_entry_v1().
    let source_fp = match source.as_ref() {
        Some(s) => source_fingerprint_v0(s)?,
        None => sha256_postcard(&"root_source_v0")?,
    };

    let dataset_fp = dataset_fingerprint_v0(source_fp, schema_fp, recipe_hash)?;

    Ok(DatasetEntryV1 {
        asset_key,
        fingerprint_v0: hex_lower(&dataset_fp),
        source_fingerprint_v0: hex_lower(&source_fp),
        schema_hash_v0: hex_lower(&schema_fp),
        recipe_hash_v0: hex_lower(&recipe_hash),
        trust,
        source,
        schema,
        license_flags: Vec::new(),
        pii_tags: Vec::new(),
    })
}

/// Build a registry entry for a derived (non-source) dataset.
///
/// Uses `derived_source_fingerprint_v0(asset_key)` to salt the source fingerprint,
/// preventing collision when multiple outputs share the same schema.
pub fn derived_dataset_entry_v1(
    asset_key: impl Into<String>,
    trust: TrustClass,
    schema: Option<SchemaDescriptorV0>,
    recipe_hash: [u8; 32],
) -> Result<DatasetEntryV1, postcard::Error> {
    let asset_key = asset_key.into();

    let source_fp = derived_source_fingerprint_v0(&asset_key)?;
    let schema_fp = match schema.as_ref() {
        Some(s) => schema_hash_v0(s)?,
        None => no_schema_hash_v0()?,
    };
    let dataset_fp = dataset_fingerprint_v0(source_fp, schema_fp, recipe_hash)?;

    Ok(DatasetEntryV1 {
        asset_key,
        fingerprint_v0: hex_lower(&dataset_fp),
        source_fingerprint_v0: hex_lower(&source_fp),
        schema_hash_v0: hex_lower(&schema_fp),
        recipe_hash_v0: hex_lower(&recipe_hash),
        trust,
        source: None,
        schema,
        license_flags: Vec::new(),
        pii_tags: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Cache-hit prediction (pure, deterministic — no registry awareness)
// ---------------------------------------------------------------------------

/// Output specification for fingerprint prediction (alpha.6+).
///
/// Accepts optional schema so prediction is exact when schema is known,
/// and falls back to `no_schema_hash_v0()` when it is not.
#[derive(Debug, Clone)]
pub struct OutputSpecCore {
    pub asset_key: String,
    pub schema: Option<SchemaDescriptorV0>,
}

/// Predicted output fingerprint (result of `predict_output_fingerprints`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictedOutput {
    pub asset_key: String,
    pub fingerprint_v0: String,
}

/// Predict output fingerprints from node definition + upstream fingerprints.
///
/// This is a **pure, deterministic** function with no registry awareness.
/// The caller (e.g. `DataOpsSession::predict`) is responsible for gathering
/// upstream fingerprints from the registry and failing closed on missing inputs.
///
/// For each output:
/// - `source_fp` = `derived_source_fingerprint_v0(asset_key)`
/// - `schema_fp` = `schema.map(schema_hash_v0).unwrap_or(no_schema_hash_v0())`
/// - `recipe` = `recipe_hash_v0(node, upstream_fps)`
/// - `fingerprint` = `dataset_fingerprint_v0(source_fp, schema_fp, recipe)`
pub fn predict_output_fingerprints(
    node: &NodeV1,
    outputs: &[OutputSpecCore],
    upstream_fps: &[[u8; 32]],
) -> Result<Vec<PredictedOutput>, postcard::Error> {
    let recipe = recipe_hash_v0(node, upstream_fps)?;

    let mut results = Vec::with_capacity(outputs.len());
    for out in outputs {
        let source_fp = derived_source_fingerprint_v0(&out.asset_key)?;
        let schema_fp = match out.schema.as_ref() {
            Some(s) => schema_hash_v0(s)?,
            None => no_schema_hash_v0()?,
        };
        let fp = dataset_fingerprint_v0(source_fp, schema_fp, recipe)?;
        results.push(PredictedOutput {
            asset_key: out.asset_key.clone(),
            fingerprint_v0: hex_lower(&fp),
        });
    }
    Ok(results)
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct CacheKeyCanonicalV0 {
    node_def_hash: [u8; 32],
    upstream_fingerprints: Vec<[u8; 32]>,
    execution_profile: String,
}

/// Deterministic cache key for materialization reuse checks.
pub fn cache_key_v0(
    node: &NodeV1,
    upstream_fps: &[[u8; 32]],
    execution_profile: &str,
) -> Result<String, postcard::Error> {
    let canonical = CacheKeyCanonicalV0 {
        node_def_hash: node_def_hash_v1(node)?,
        upstream_fingerprints: upstream_fps.to_vec(),
        execution_profile: normalize_lower(execution_profile),
    };
    Ok(hex_lower(&sha256_postcard(&canonical)?))
}

/// Compatibility helper for deriving `cache_hit` from a decision enum.
pub fn cache_hit_from_decision(decision: CacheDecisionV0) -> Option<bool> {
    match decision {
        CacheDecisionV0::Hit => Some(true),
        CacheDecisionV0::Miss => Some(false),
        CacheDecisionV0::Bypass | CacheDecisionV0::Unknown => None,
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
            execution_hint: None,
            cache_policy: None,
            materialization_policy: None,
            resources: None,
            op_hash: None,
        };

        let upstream = [[7u8; 32]];
        let recipe = recipe_hash_v0(&node, &upstream).unwrap();

        let a = dataset_entry_v1(
            "dataset://ns/clean",
            TrustClass::Trusted,
            Some(source.clone()),
            Some(schema.clone()),
            recipe,
        )
        .unwrap();
        let b = dataset_entry_v1(
            "dataset://ns/clean",
            TrustClass::Trusted,
            Some(source),
            Some(schema),
            recipe,
        )
        .unwrap();
        assert_eq!(a.fingerprint_v0, b.fingerprint_v0);
        assert_eq!(a.schema_hash_v0, b.schema_hash_v0);
        assert_eq!(a.source_fingerprint_v0, b.source_fingerprint_v0);
    }

    #[test]
    fn canonical_placeholder_no_schema_is_deterministic() {
        let a = no_schema_hash_v0().unwrap();
        let b = no_schema_hash_v0().unwrap();
        assert_eq!(a, b, "no_schema_hash_v0 must be deterministic");
        // Must be 32 bytes
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn canonical_placeholder_derived_is_salted() {
        let a = derived_source_fingerprint_v0("dataset://ns/left").unwrap();
        let b = derived_source_fingerprint_v0("dataset://ns/right").unwrap();
        assert_ne!(
            a, b,
            "different asset_keys must produce different fingerprints"
        );
        // Same asset_key is deterministic
        let a2 = derived_source_fingerprint_v0("dataset://ns/left").unwrap();
        assert_eq!(a, a2, "same asset_key must produce same fingerprint");
    }

    #[test]
    fn derived_dataset_entry_uses_canonical_helpers() {
        let recipe = [42u8; 32];
        let entry = derived_dataset_entry_v1("dataset://ns/out", TrustClass::Trusted, None, recipe)
            .unwrap();

        // Check that source_fingerprint uses derived_source_fingerprint_v0
        let expected_source_fp = derived_source_fingerprint_v0("dataset://ns/out").unwrap();
        assert_eq!(entry.source_fingerprint_v0, hex_lower(&expected_source_fp));

        // Check that schema_hash uses no_schema_hash_v0
        let expected_schema_hash = no_schema_hash_v0().unwrap();
        assert_eq!(entry.schema_hash_v0, hex_lower(&expected_schema_hash));
    }

    #[test]
    fn materialization_v2_serialization_roundtrip() {
        let node_id = NodeId::from_bytes([9u8; 16]);
        let row = MaterializationRecordV2 {
            schema_version: MATERIALIZATION_SCHEMA_V2,
            record_seq: 17,
            ts_unix_nanos: 1_234_567,
            asset_key: "dataset://ns/out".to_string(),
            fingerprint_v0: "a".repeat(64),
            node_id,
            node_def_hash: "b".repeat(64),
            op_type: "filter_rows".to_string(),
            input_asset_keys: vec!["dataset://ns/in".to_string()],
            input_fingerprints_v0: vec!["c".repeat(64)],
            rows: Some(10),
            bytes: Some(20),
            duration_ms: Some(5),
            cache_decision: CacheDecisionV0::Hit,
            cache_reason: Some("cache key match".to_string()),
            cache_key_v0: Some("d".repeat(64)),
            cache_hit: cache_hit_from_decision(CacheDecisionV0::Hit),
            unsafe_surface: false,
            unsafe_reasons: Vec::new(),
            applied_transforms: Vec::new(),
            status: MaterializationStatusV0::Ok,
            error_code: None,
            quality: Some(QualitySummaryV0 {
                null_rate: Some(0.0),
                row_count_delta: Some(0),
                schema_changed: Some(false),
            }),
        };

        let json = serde_json::to_string(&row).expect("serialize v2");
        let decoded: MaterializationRecordV2 = serde_json::from_str(&json).expect("deserialize v2");
        assert_eq!(decoded, row);
    }

    #[test]
    fn transform_audit_v0_serialization_roundtrip() {
        let audit = TransformAuditV0 {
            transform_name: "dp_clip".to_string(),
            core_trusted: false,
            round_id: 7,
        };
        let json = serde_json::to_string(&audit).unwrap();
        let decoded: TransformAuditV0 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, audit);
    }

    #[test]
    fn applied_transforms_empty_by_default_in_record() {
        let json = r#"{
            "schema_version": 2,
            "record_seq": 1,
            "ts_unix_nanos": 100,
            "asset_key": "dataset://ns/out",
            "fingerprint_v0": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "node_id": "01010101010101010101010101010101",
            "node_def_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "op_type": "transform",
            "cache_decision": "unknown",
            "unsafe_surface": false,
            "status": "ok"
        }"#;
        let decoded: MaterializationRecordV2 = serde_json::from_str(json).unwrap();
        assert!(
            decoded.applied_transforms.is_empty(),
            "missing field must default to empty"
        );
    }

    #[test]
    fn source_descriptor_redacts_userinfo_in_uri() {
        let source_a = SourceDescriptorV0 {
            uri: "s3://alice:secret@bucket/path/file.parquet?part=1".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::BearerToken,
            etag_or_version: Some("v1".to_string()),
        };
        let source_b = SourceDescriptorV0 {
            uri: "s3://bob:other-secret@bucket/path/file.parquet?part=1".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::BearerToken,
            etag_or_version: Some("v1".to_string()),
        };

        let sanitized = sanitize_source_descriptor_v0(&source_a).expect("sanitize should succeed");
        assert_eq!(sanitized.uri, "s3://<redacted>@bucket/path/file.parquet");

        // Fingerprints must ignore differing credentials after redaction.
        assert_eq!(
            source_fingerprint_v0(&source_a),
            source_fingerprint_v0(&source_b),
            "userinfo changes should not affect source fingerprint"
        );
    }

    #[test]
    fn source_descriptor_strips_query_and_fragment() {
        let with_query = SourceDescriptorV0 {
            uri: "s3://bucket/path/file.parquet?version=2&token=abc".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: None,
        };
        let with_fragment = SourceDescriptorV0 {
            uri: "s3://bucket/path/file.parquet#sheet1".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: None,
        };
        let clean = SourceDescriptorV0 {
            uri: "s3://bucket/path/file.parquet".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: None,
        };

        let sanitized_q = sanitize_source_descriptor_v0(&with_query).unwrap();
        let sanitized_f = sanitize_source_descriptor_v0(&with_fragment).unwrap();
        let sanitized_c = sanitize_source_descriptor_v0(&clean).unwrap();

        assert_eq!(sanitized_q.uri, "s3://bucket/path/file.parquet");
        assert_eq!(sanitized_f.uri, "s3://bucket/path/file.parquet");
        assert_eq!(sanitized_c.uri, "s3://bucket/path/file.parquet");

        // All three must produce the same fingerprint
        let fp_q = source_fingerprint_v0(&with_query).unwrap();
        let fp_f = source_fingerprint_v0(&with_fragment).unwrap();
        let fp_c = source_fingerprint_v0(&clean).unwrap();
        assert_eq!(fp_q, fp_c, "query should not affect fingerprint");
        assert_eq!(fp_f, fp_c, "fragment should not affect fingerprint");
    }

    #[test]
    fn source_descriptor_rejects_oversized_uri() {
        let source = SourceDescriptorV0 {
            uri: format!("s3://bucket/{}", "a".repeat(MAX_SOURCE_URI_LEN + 1)),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: None,
        };

        let result = sanitize_source_descriptor_v0(&source);
        assert!(matches!(
            result,
            Err(SourceDescriptorError::UriTooLong { .. })
        ));
    }

    #[test]
    fn source_descriptor_rejects_oversized_etag_or_version() {
        let source = SourceDescriptorV0 {
            uri: "s3://bucket/path".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: Some("e".repeat(MAX_ETAG_OR_VERSION_LEN + 1)),
        };

        let result = sanitize_source_descriptor_v0(&source);
        assert!(matches!(
            result,
            Err(SourceDescriptorError::EtagOrVersionTooLong { .. })
        ));
    }

    #[test]
    fn unsafe_reasons_include_missing_provenance() {
        let legacy = MaterializationRecordV1 {
            schema_version: 1,
            ts_unix_nanos: 42,
            asset_key: "dataset://ns/v1".to_string(),
            fingerprint_v0: "a".repeat(64),
            node_id: NodeId::from_bytes([1u8; 16]),
            node_def_hash: "b".repeat(64),
            rows: Some(1),
            bytes: Some(2),
            cache_hit: Some(true),
            duration_ms: Some(3),
            quality_flags: None,
            unsafe_surface: false,
        };

        let normalized = MaterializationRecordCompat::V1(legacy).into_v2();
        assert!(
            normalized
                .unsafe_reasons
                .contains(&UnsafeReasonV0::MissingProvenance),
            "legacy compatibility normalization should mark missing provenance"
        );
        assert!(
            normalized.unsafe_surface,
            "missing provenance reason should mark record unsafe"
        );
    }

    // ── M-12: Read-path descriptor bounds validation ──

    #[test]
    fn validate_bounds_rejects_oversized_uri_on_read() {
        let desc = SourceDescriptorV0 {
            uri: "x".repeat(MAX_SOURCE_URI_LEN + 1),
            content_type: "text/plain".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: None,
        };
        assert!(matches!(
            validate_source_descriptor_bounds(&desc),
            Err(SourceDescriptorError::UriTooLong { .. })
        ));
    }

    #[test]
    fn validate_bounds_accepts_within_limit_descriptor() {
        let desc = SourceDescriptorV0 {
            uri: "s3://bucket/path".to_string(),
            content_type: "application/parquet".to_string(),
            auth_mode: AuthModeMarker::None,
            etag_or_version: Some("v1".to_string()),
        };
        assert!(validate_source_descriptor_bounds(&desc).is_ok());
    }
}
