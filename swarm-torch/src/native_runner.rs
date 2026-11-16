//! Minimal native OpRunner (alpha.6, std-only).
//!
//! Implements core dataops operators (`passthrough`, `filter_rows`, `union`) plus
//! policy-pinned `wasm_transform` dispatch over attached external executors:
//! - Physical CAS execution (`RunnerExecutionModeV1::PhysicalStrict`): reads/writes verified CAS blobs
//!   with real `rows`/`bytes` measurements.
//! - Explicit test metadata mode (`RunnerExecutionModeV1::MetadataOnlyForTests`): metadata-only fallback.
//!
//! All ops emit a deterministic span:
//! - `trace_id = run_id` (16 bytes → TraceId)
//! - `span_id = sha256(node_id_bytes || ts_nanos_be)[0..8]`
//!
//! **ADR-0018:** The runner boundary is separate from the scheduler.
//! Policy enforcement must happen BEFORE calling `run()`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use swarm_torch_core::execution::{AssetInstanceV1, OpRunner};
use swarm_torch_core::observe::{AttrMap, RunEventEmitter, RunId, SpanId, SpanRecord, TraceId};
use swarm_torch_core::run_graph::{CanonParams, CanonValue, ExecutionTrust, NodeV1};

use crate::artifacts::schema_check::{
    validate_json_bytes_against_schema, COLUMNAR_TABLE_SCHEMA_JSON,
};
use crate::artifacts::{
    cmp_json_numbers, hex_lower, parse_cas_uri_digest, parse_finite_json_number_str,
    read_bytes_bounded, write_once_cas_blob, OutputSpec, MAX_NDJSON_LINES, MAX_NDJSON_LINE_BYTES,
    MAX_SINGLE_ARTIFACT_BYTES,
};

/// Maximum number of rows permitted in a validated [`ColumnarTableV1`] (`ST-15`).
pub const MAX_COLUMNAR_ROWS: usize = MAX_NDJSON_LINES;
/// Maximum number of columns permitted in a validated [`ColumnarTableV1`] (`ST-15`).
pub const MAX_COLUMNAR_COLUMNS: usize = 4_096;
use crate::wasm_sandbox::{
    resolve_and_preflight_wasm_node_v1, wasm_error_to_io, SandboxedExtensionPolicyV1,
    WasmExtensionExecutor, WasmSandboxError,
};

/// Canonical column data type discriminator matching `schemas/v1/columnar_table.schema.json` (`P2-20b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnDtypeV1 {
    #[serde(rename = "i64")]
    I64,
    #[serde(rename = "f64")]
    F64,
    #[serde(rename = "bool")]
    Bool,
    #[serde(rename = "utf8")]
    Utf8,
    #[serde(rename = "bytes_hex")]
    BytesHex,
}

/// Column definition inside [`ColumnarTableV1`] (`P2-20b`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnSchemaV1 {
    /// Column name (`minLength >= 1`).
    pub name: String,
    /// Column storage type.
    pub dtype: ColumnDtypeV1,
    /// Whether `null` cell entries are permitted.
    pub nullable: bool,
}

/// Canonical deterministic columnar table (`stcl-json-v1`, `P2-20b` / `W12-04`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnarTableV1 {
    /// Fixed schema version `1`.
    pub schema_version: u32,
    /// Fixed table format tag `"stcl-json-v1"`.
    pub table_format: String,
    /// Number of rows (`rows.len() as u64`).
    pub row_count: u64,
    /// Ordered column definitions.
    pub columns: Vec<ColumnSchemaV1>,
    /// Row-major cell values (`Value::Number`, `Bool`, `String`, or `Null`).
    pub rows: Vec<Vec<Value>>,
    /// Canonical SHA-256 hex digest of deterministic rows + column header (`stcl-v1`), optional on wire.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_digest_sha256: Option<String>,
}

impl ColumnarTableV1 {
    /// Construct and strictly validate a [`ColumnarTableV1`] against `columns`, checking nullability,
    /// finiteness, and column types.
    pub fn try_new(columns: Vec<ColumnSchemaV1>, rows: Vec<Vec<Value>>) -> io::Result<Self> {
        let mut table = Self {
            schema_version: 1,
            table_format: "stcl-json-v1".to_string(),
            row_count: rows.len() as u64,
            columns,
            rows,
            content_digest_sha256: None,
        };
        table.validate()?;
        table.content_digest_sha256 = Some(Self::compute_content_digest_sha256(
            &table.columns,
            &table.rows,
        )?);
        Ok(table)
    }

    fn compute_content_digest_sha256(
        columns: &[ColumnSchemaV1],
        rows: &[Vec<Value>],
    ) -> io::Result<String> {
        let body_bytes = serde_json::to_vec(&(columns, rows))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let digest = Sha256::digest(&body_bytes);
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&digest[..]);
        Ok(hex_lower(&arr))
    }

    /// Parse and validate a [`ColumnarTableV1`] from raw JSON bytes against `COLUMNAR_TABLE_SCHEMA_JSON`.
    pub fn from_json_bytes_strict(bytes: &[u8]) -> io::Result<Self> {
        validate_json_bytes_against_schema(bytes, COLUMNAR_TABLE_SCHEMA_JSON).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("columnar schema violation: {e}"),
            )
        })?;
        let table: Self = serde_json::from_slice(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        table.validate()?;
        Ok(table)
    }

    /// Serialize to canonical JSON bytes verified against `COLUMNAR_TABLE_SCHEMA_JSON`.
    pub fn to_canonical_json_bytes(&self) -> io::Result<Vec<u8>> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        validate_json_bytes_against_schema(&bytes, COLUMNAR_TABLE_SCHEMA_JSON).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("columnar schema violation: {e}"),
            )
        })?;
        Ok(bytes)
    }

    /// Validate schema constraints, row width, nullability, and value dtypes.
    pub fn validate(&self) -> io::Result<()> {
        if self.schema_version != 1 || self.table_format != "stcl-json-v1" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid ColumnarTableV1 schema_version or table_format",
            ));
        }
        if self.columns.len() > MAX_COLUMNAR_COLUMNS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ColumnarTableV1 column count {} exceeds MAX_COLUMNAR_COLUMNS ({MAX_COLUMNAR_COLUMNS})",
                    self.columns.len()
                ),
            ));
        }
        if self.rows.len() > MAX_COLUMNAR_ROWS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "ColumnarTableV1 row count {} exceeds MAX_COLUMNAR_ROWS ({MAX_COLUMNAR_ROWS})",
                    self.rows.len()
                ),
            ));
        }
        if self.row_count != self.rows.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ColumnarTableV1 row_count does not match rows.len()",
            ));
        }
        let mut seen_columns = BTreeSet::new();
        for (col_idx, col) in self.columns.iter().enumerate() {
            if col.name.is_empty() || col.name.chars().count() > 256 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid column name length at index {col_idx}"),
                ));
            }
            if !seen_columns.insert(col.name.as_str()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("duplicate column name '{}'", col.name),
                ));
            }
        }
        for (row_idx, row) in self.rows.iter().enumerate() {
            if row.len() != self.columns.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "ColumnarTableV1 row {row_idx} width {} != {}",
                        row.len(),
                        self.columns.len()
                    ),
                ));
            }
            for (col_idx, cell) in row.iter().enumerate() {
                let col = &self.columns[col_idx];
                if cell.is_null() {
                    if !col.nullable {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("null value in non-nullable column '{}'", col.name),
                        ));
                    }
                    continue;
                }
                let ok = match col.dtype {
                    ColumnDtypeV1::I64 => cell.as_i64().is_some(),
                    ColumnDtypeV1::F64 => cell.as_f64().is_some_and(|v| v.is_finite()),
                    ColumnDtypeV1::Bool => cell.is_boolean(),
                    ColumnDtypeV1::Utf8 => cell.is_string(),
                    ColumnDtypeV1::BytesHex => cell.as_str().is_some_and(|s| {
                        s.len() % 2 == 0 && s.bytes().all(|b| b.is_ascii_hexdigit())
                    }),
                };
                if !ok {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "cell ({row_idx}, {}) type mismatch for column '{}' ({:?})",
                            col_idx, col.name, col.dtype
                        ),
                    ));
                }
            }
        }
        if let Some(ref expected_digest) = self.content_digest_sha256 {
            let actual_digest = Self::compute_content_digest_sha256(&self.columns, &self.rows)?;
            if &actual_digest != expected_digest {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ColumnarTableV1 content_digest_sha256 mismatch",
                ));
            }
        }
        Ok(())
    }
}

/// Borrowed Wasm sandbox policy + module registry context passed to [`NativeOpRunner::run_in_session`].
pub type WasmIsolationCtx<'a> = (
    &'a SandboxedExtensionPolicyV1,
    &'a BTreeMap<[u8; 32], Vec<u8>>,
);

/// Canonical `content_type` for `ColumnarTableV1` JSON payloads (`ST-10`).
pub const CONTENT_TYPE_COLUMNAR_JSON_V1: &str = "application/vnd.swarmtorch.columnar+json";
/// Canonical `content_type` for line-delimited JSON object streams (`ST-10`).
pub const CONTENT_TYPE_NDJSON: &str = "application/x-ndjson";
/// Canonical `content_type` for opaque binary CAS payloads (such as model checkpoints).
pub const CONTENT_TYPE_OCTET_STREAM: &str = "application/octet-stream";

/// Declared physical payload format resolved from `SourceDescriptorV0.content_type` (`ST-10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnarInputFormatV1 {
    ColumnarTableV1,
    Ndjson,
    OpaqueBinary,
}

impl ColumnarInputFormatV1 {
    /// Parse a declared `SourceDescriptorV0.content_type` without falling back to byte sniffing.
    pub fn from_content_type(content_type: &str) -> io::Result<Self> {
        let mime = content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match mime.as_str() {
            CONTENT_TYPE_COLUMNAR_JSON_V1 => Ok(Self::ColumnarTableV1),
            CONTENT_TYPE_NDJSON | "application/ndjson" => Ok(Self::Ndjson),
            CONTENT_TYPE_OCTET_STREAM => Ok(Self::OpaqueBinary),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unsupported or ambiguous content_type '{other}'; expected '{CONTENT_TYPE_COLUMNAR_JSON_V1}', '{CONTENT_TYPE_NDJSON}', or '{CONTENT_TYPE_OCTET_STREAM}'"
                ),
            )),
        }
    }

    /// Canonical MIME `content_type` string emitted for outputs of this format.
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::ColumnarTableV1 => CONTENT_TYPE_COLUMNAR_JSON_V1,
            Self::Ndjson => CONTENT_TYPE_NDJSON,
            Self::OpaqueBinary => CONTENT_TYPE_OCTET_STREAM,
        }
    }
}

/// Execution strictness mode for [`NativeOpRunner`] (`ST-10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunnerExecutionModeV1 {
    /// Fail closed unless every declared input resolves to an authenticated `cas://sha256/`
    /// blob with an explicitly declared `content_type`. Default for production & SDK callers.
    #[default]
    PhysicalStrict,
    /// Explicit metadata-only fallback permitted solely for unit/scheduler tests without CAS blobs.
    MetadataOnlyForTests,
}

/// Output produced by [`NativeOpRunner::run_in_session_with_content_types`], preserving physical
/// row/byte counts, declared output content-types, and CAS URIs alongside [`AssetInstanceV1`] descriptors.
#[derive(Debug, Clone)]
pub struct PhysicalExecOutcomeV1 {
    /// Emitted output asset instances (including `cas://sha256/<hex>` URIs).
    pub instances: Vec<AssetInstanceV1>,
    /// Computed output specs carrying real physical row/byte measurements.
    pub output_specs: Vec<OutputSpec>,
    /// Declared canonical `content_type` keyed by output `asset_key`.
    pub output_content_types: BTreeMap<String, String>,
}

/// Execution context for the native runner.
///
/// Provides `run_id` (→ trace_id) and a clock function for span timestamps.
/// The clock function allows deterministic testing.
pub struct ExecutionContext {
    pub run_id: RunId,
    pub clock_nanos: fn() -> u64,
}

/// Deterministic span ID: `sha256(node_id_bytes || ts_nanos_be)[0..8]`.
fn deterministic_span_id(node_id_bytes: &[u8; 16], ts_nanos: u64) -> SpanId {
    let mut hasher = Sha256::new();
    hasher.update(node_id_bytes);
    hasher.update(ts_nanos.to_be_bytes());
    let hash = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash[..8]);
    SpanId::from_bytes(bytes)
}

/// Native DataOps [`OpRunner`] supporting physical CAS execution and explicit test metadata mode.
///
/// Defaults to [`RunnerExecutionModeV1::PhysicalStrict`] (`ST-10`).
///
/// Supports three `op_type`s:
/// - `"passthrough"` — validates/copies CAS inputs (or forwards metadata in [`RunnerExecutionModeV1::MetadataOnlyForTests`])
/// - `"filter_rows"` — executes typed row filtering over `ColumnarTableV1` or NDJSON CAS blobs
/// - `"union"` — merges homogeneous `ColumnarTableV1` or NDJSON CAS inputs
#[derive(Clone, Default)]
pub struct NativeOpRunner {
    mode: RunnerExecutionModeV1,
    wasm_executor: Option<Arc<dyn WasmExtensionExecutor>>,
}

impl fmt::Debug for NativeOpRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeOpRunner")
            .field("mode", &self.mode)
            .field("has_wasm_executor", &self.wasm_executor.is_some())
            .finish()
    }
}

impl NativeOpRunner {
    /// Construct a fail-closed [`RunnerExecutionModeV1::PhysicalStrict`] runner.
    pub const fn new() -> Self {
        Self {
            mode: RunnerExecutionModeV1::PhysicalStrict,
            wasm_executor: None,
        }
    }

    /// Construct a runner with an explicit [`RunnerExecutionModeV1`].
    pub const fn with_mode(mode: RunnerExecutionModeV1) -> Self {
        Self {
            mode,
            wasm_executor: None,
        }
    }

    /// Attach an isolated external WebAssembly extension executor (`ST-09`).
    pub fn with_wasm_executor(mut self, executor: Arc<dyn WasmExtensionExecutor>) -> Self {
        self.wasm_executor = Some(executor);
        self
    }

    /// Return the configured external WebAssembly extension executor, if any.
    pub fn wasm_executor(&self) -> Option<&Arc<dyn WasmExtensionExecutor>> {
        self.wasm_executor.as_ref()
    }

    /// Return the configured [`RunnerExecutionModeV1`].
    pub const fn mode(&self) -> RunnerExecutionModeV1 {
        self.mode
    }

    /// Run with explicit execution context.
    ///
    /// Fails closed under [`RunnerExecutionModeV1::PhysicalStrict`] because no `run_dir`/CAS
    /// session is available; use [`NativeOpRunner::run_in_session_with_content_types`] or
    /// [`NativeOpRunner::with_mode`] with [`RunnerExecutionModeV1::MetadataOnlyForTests`].
    pub fn run_with_context<E: RunEventEmitter<Error = io::Error>>(
        &self,
        ctx: &ExecutionContext,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        emitter: &E,
    ) -> io::Result<Vec<AssetInstanceV1>> {
        if self.mode == RunnerExecutionModeV1::PhysicalStrict {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "PhysicalStrict NativeOpRunner requires a CAS session; use run_in_session_with_content_types or RunnerExecutionModeV1::MetadataOnlyForTests",
            ));
        }

        let start_nanos = (ctx.clock_nanos)();

        // Resolve node_id
        let node_id = node
            .node_id
            .unwrap_or_else(|| swarm_torch_core::run_graph::node_id_from_key(&node.node_key));
        let node_id_bytes = node_id.as_bytes();

        let span_id = deterministic_span_id(node_id_bytes, start_nanos);
        let trace_id = TraceId::from_bytes(*ctx.run_id.as_bytes());

        // Dispatch by op_type
        let outputs = match node.op_type.as_str() {
            "passthrough" => Self::op_passthrough(inputs),
            "filter_rows" => Self::op_filter_rows(inputs, node),
            "union" => Self::op_union(inputs, node),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsupported op_type: {}", other),
                ));
            }
        };

        let end_nanos = (ctx.clock_nanos)();
        Self::emit_op_span(
            trace_id,
            span_id,
            node,
            inputs.len(),
            outputs.len(),
            (start_nanos, end_nanos),
            emitter,
        )?;

        Ok(outputs)
    }

    fn emit_op_span<E: RunEventEmitter<Error = io::Error>>(
        trace_id: TraceId,
        span_id: SpanId,
        node: &NodeV1,
        input_count: usize,
        output_count: usize,
        timings_unix_nanos: (u64, u64),
        emitter: &E,
    ) -> io::Result<()> {
        let (start_unix_nanos, end_unix_nanos) = timings_unix_nanos;
        let mut attrs: AttrMap = BTreeMap::new();
        attrs.insert(
            "swarmtorch.op_type".to_string(),
            swarm_torch_core::observe::AttrValue::Str(node.op_type.clone()),
        );
        attrs.insert(
            "swarmtorch.node_key".to_string(),
            swarm_torch_core::observe::AttrValue::Str(node.node_key.clone()),
        );
        attrs.insert(
            "swarmtorch.input_count".to_string(),
            swarm_torch_core::observe::AttrValue::I64(input_count as i64),
        );
        attrs.insert(
            "swarmtorch.output_count".to_string(),
            swarm_torch_core::observe::AttrValue::I64(output_count as i64),
        );

        let span = SpanRecord {
            schema_version: 1,
            trace_id,
            span_id,
            parent_span_id: None,
            name: format!("op/{}", node.op_type),
            start_unix_nanos,
            end_unix_nanos: Some(end_unix_nanos),
            attrs,
        };
        emitter.emit_span(&span)
    }

    /// Execute a graph operator against physical CAS blobs in `run_dir` without additional
    /// declared input `content_type`s.
    ///
    /// Under [`RunnerExecutionModeV1::PhysicalStrict`], physical operators fail closed unless
    /// declared input `content_type`s are supplied via [`Self::run_in_session_with_content_types`].
    pub fn run_in_session<E: RunEventEmitter<Error = io::Error>>(
        &self,
        ctx: &ExecutionContext,
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        emitter: &E,
        wasm_isolation: Option<WasmIsolationCtx<'_>>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let empty = BTreeMap::new();
        self.run_in_session_with_content_types(
            ctx,
            run_dir,
            node,
            inputs,
            emitter,
            wasm_isolation,
            &empty,
        )
    }

    /// Execute a graph operator against physical CAS blobs in `run_dir` using explicit declared
    /// `input_content_types` (`asset_key -> SourceDescriptorV0.content_type`, `ST-10`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_in_session_with_content_types<E: RunEventEmitter<Error = io::Error>>(
        &self,
        ctx: &ExecutionContext,
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        emitter: &E,
        wasm_isolation: Option<WasmIsolationCtx<'_>>,
        input_content_types: &BTreeMap<String, String>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let start_nanos = (ctx.clock_nanos)();
        let node_id = node
            .node_id
            .unwrap_or_else(|| swarm_torch_core::run_graph::node_id_from_key(&node.node_key));
        let span_id = deterministic_span_id(node_id.as_bytes(), start_nanos);
        let trace_id = TraceId::from_bytes(*ctx.run_id.as_bytes());

        if node.op_type == "wasm_transform" {
            let outcome = self.physical_wasm_transform(
                run_dir,
                node,
                inputs,
                wasm_isolation,
                input_content_types,
            )?;
            let end_nanos = (ctx.clock_nanos)();
            Self::emit_op_span(
                trace_id,
                span_id,
                node,
                inputs.len(),
                outcome.instances.len(),
                (start_nanos, end_nanos),
                emitter,
            )?;
            return Ok(outcome);
        }

        if node.execution_trust != ExecutionTrust::Core {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "native op '{}' requires ExecutionTrust::Core, got {:?}",
                    node.op_type, node.execution_trust
                ),
            ));
        }

        if !matches!(
            node.op_type.as_str(),
            "passthrough" | "filter_rows" | "union"
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported op_type: {}", node.op_type),
            ));
        }

        let has_cas_inputs = !inputs.is_empty()
            && inputs
                .iter()
                .all(|i| i.uri.as_deref().and_then(parse_cas_uri_digest).is_some());

        if !has_cas_inputs {
            match self.mode {
                RunnerExecutionModeV1::PhysicalStrict => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "PhysicalStrict NativeOpRunner requires non-empty cas://sha256 inputs for '{}'",
                            node.op_type
                        ),
                    ));
                }
                RunnerExecutionModeV1::MetadataOnlyForTests => {
                    let instances = self.run_with_context(ctx, node, inputs, emitter)?;
                    let output_specs = node
                        .outputs
                        .iter()
                        .map(|o| OutputSpec {
                            asset_key: o.asset_key.clone(),
                            schema: None,
                            rows: None,
                            bytes: None,
                        })
                        .collect();
                    return Ok(PhysicalExecOutcomeV1 {
                        instances,
                        output_specs,
                        output_content_types: BTreeMap::new(),
                    });
                }
            }
        }

        // Physical CAS execution (`passthrough`, `filter_rows`, `union`) with strict content-type dispatch.
        let outcome = match node.op_type.as_str() {
            "passthrough" => {
                Self::physical_passthrough(run_dir, node, inputs, input_content_types)?
            }
            "filter_rows" => {
                Self::physical_filter_rows(run_dir, node, inputs, input_content_types)?
            }
            "union" => Self::physical_union(run_dir, node, inputs, input_content_types)?,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsupported op_type: {other}"),
                ));
            }
        };

        let end_nanos = (ctx.clock_nanos)();
        Self::emit_op_span(
            trace_id,
            span_id,
            node,
            inputs.len(),
            outcome.instances.len(),
            (start_nanos, end_nanos),
            emitter,
        )?;
        Ok(outcome)
    }

    fn resolve_input_format(
        instance: &AssetInstanceV1,
        input_content_types: &BTreeMap<String, String>,
    ) -> io::Result<ColumnarInputFormatV1> {
        let ct = input_content_types
            .get(&instance.asset_key)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "missing declared SourceDescriptorV0.content_type for physical input '{}'",
                        instance.asset_key
                    ),
                )
            })?;
        ColumnarInputFormatV1::from_content_type(ct)
    }

    fn parse_ndjson_rows(raw: &[u8]) -> io::Result<Vec<(String, Value)>> {
        for raw_line in raw.split(|&b| b == b'\n') {
            let line_bytes = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
            if line_bytes.len() > MAX_NDJSON_LINE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "NDJSON line length {} exceeds maximum {MAX_NDJSON_LINE_BYTES}",
                        line_bytes.len()
                    ),
                ));
            }
        }
        let text = std::str::from_utf8(raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let mut parsed = Vec::new();
        for (line_idx, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if parsed.len() >= MAX_NDJSON_LINES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("NDJSON row count exceeds maximum {MAX_NDJSON_LINES}"),
                ));
            }
            let obj: Value = serde_json::from_str(trimmed)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            if !obj.is_object() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("NDJSON row {} must be a JSON object", line_idx + 1),
                ));
            }
            parsed.push((trimmed.to_string(), obj));
        }
        Ok(parsed)
    }

    fn validate_declared_output_keys(
        node: &NodeV1,
        expected_len: usize,
        op_name: &str,
    ) -> io::Result<Vec<String>> {
        if node.outputs.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{op_name} requires {expected_len} declared output(s), got {}",
                    node.outputs.len()
                ),
            ));
        }
        let mut seen = BTreeSet::new();
        let mut keys = Vec::with_capacity(expected_len);
        for out in &node.outputs {
            if out.asset_key.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{op_name} declared output has empty asset_key"),
                ));
            }
            if !seen.insert(out.asset_key.as_str()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "{op_name} declared duplicate output asset_key '{}'",
                        out.asset_key
                    ),
                ));
            }
            keys.push(out.asset_key.clone());
        }
        Ok(keys)
    }

    fn commit_staged_cas_outputs(
        run_dir: &Path,
        staged: Vec<(String, Vec<u8>, Option<u64>, ColumnarInputFormatV1)>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let mut instances = Vec::with_capacity(staged.len());
        let mut output_specs = Vec::with_capacity(staged.len());
        let mut output_content_types = BTreeMap::new();
        for (out_key, bytes, row_count, fmt) in staged {
            let (hex, uri, byte_len) = write_cas_bytes(run_dir, &bytes)?;
            instances.push(AssetInstanceV1 {
                asset_key: out_key.clone(),
                fingerprint_v0: hex,
                uri: Some(uri),
            });
            output_specs.push(OutputSpec {
                asset_key: out_key.clone(),
                schema: None,
                rows: row_count,
                bytes: Some(byte_len),
            });
            output_content_types.insert(out_key, fmt.content_type().to_string());
        }
        Ok(PhysicalExecOutcomeV1 {
            instances,
            output_specs,
            output_content_types,
        })
    }

    fn physical_passthrough(
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        input_content_types: &BTreeMap<String, String>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let out_keys = Self::validate_declared_output_keys(node, inputs.len(), "passthrough")?;
        let mut staged = Vec::with_capacity(inputs.len());
        for (inp, out_key) in inputs.iter().zip(out_keys) {
            let fmt = Self::resolve_input_format(inp, input_content_types)?;
            let raw = read_verified_cas_input(run_dir, inp)?;
            let row_count = match fmt {
                ColumnarInputFormatV1::ColumnarTableV1 => {
                    let tbl = ColumnarTableV1::from_json_bytes_strict(&raw)?;
                    Some(tbl.row_count)
                }
                ColumnarInputFormatV1::Ndjson => Some(Self::parse_ndjson_rows(&raw)?.len() as u64),
                ColumnarInputFormatV1::OpaqueBinary => None,
            };
            staged.push((out_key, raw, row_count, fmt));
        }
        Self::commit_staged_cas_outputs(run_dir, staged)
    }

    fn physical_filter_rows(
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        input_content_types: &BTreeMap<String, String>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let out_keys = Self::validate_declared_output_keys(node, inputs.len(), "filter_rows")?;
        let col_name = canon_param_str(&node.params, "column");
        let raw_op = canon_param_str(&node.params, "op").unwrap_or("eq");
        let target_val = canon_param_str(&node.params, "value");
        if col_name.is_some() != target_val.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "filter_rows requires both 'column' and 'value' parameters when filtering",
            ));
        }
        let op = match raw_op {
            "eq" => "eq",
            "ne" => "ne",
            "gt" => "gt",
            "ge" | "gte" => "ge",
            "lt" => "lt",
            "le" | "lte" => "le",
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsupported filter_rows operator '{other}'"),
                ));
            }
        };

        let mut staged = Vec::with_capacity(inputs.len());
        for (inp, out_key) in inputs.iter().zip(out_keys) {
            let fmt = Self::resolve_input_format(inp, input_content_types)?;
            let raw = read_verified_cas_input(run_dir, inp)?;
            let (out_bytes, rows_kept) = match fmt {
                ColumnarInputFormatV1::ColumnarTableV1 => {
                    let table = ColumnarTableV1::from_json_bytes_strict(&raw)?;
                    let filtered_rows = if let (Some(col), Some(rhs)) = (col_name, target_val) {
                        let col_idx = table
                            .columns
                            .iter()
                            .position(|c| c.name == col)
                            .ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    format!("filter_rows column '{col}' not found in table schema"),
                                )
                            })?;
                        table
                            .rows
                            .into_iter()
                            .filter(|row| cell_matches_predicate(&row[col_idx], op, rhs))
                            .collect()
                    } else {
                        table.rows
                    };
                    let rows_kept = filtered_rows.len() as u64;
                    let filtered_table = ColumnarTableV1::try_new(table.columns, filtered_rows)?;
                    (filtered_table.to_canonical_json_bytes()?, rows_kept)
                }
                ColumnarInputFormatV1::Ndjson => {
                    let parsed = Self::parse_ndjson_rows(&raw)?;
                    let mut out_lines = Vec::new();
                    for (line, obj) in parsed {
                        let keep = match (col_name, target_val) {
                            (Some(col), Some(rhs)) => obj
                                .get(col)
                                .is_some_and(|cell| cell_matches_predicate(cell, op, rhs)),
                            _ => true,
                        };
                        if keep {
                            out_lines.push(line);
                        }
                    }
                    let count = out_lines.len() as u64;
                    let mut buf = out_lines.join("\n").into_bytes();
                    if !buf.is_empty() {
                        buf.push(b'\n');
                    }
                    (buf, count)
                }
                ColumnarInputFormatV1::OpaqueBinary => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "filter_rows requires tabular content_type ('{CONTENT_TYPE_COLUMNAR_JSON_V1}' or '{CONTENT_TYPE_NDJSON}'), got '{CONTENT_TYPE_OCTET_STREAM}' for '{}'",
                            inp.asset_key
                        ),
                    ));
                }
            };
            staged.push((out_key, out_bytes, Some(rows_kept), fmt));
        }
        Self::commit_staged_cas_outputs(run_dir, staged)
    }

    fn physical_union(
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        input_content_types: &BTreeMap<String, String>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        if inputs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "union requires at least one physical input",
            ));
        }
        let mut out_keys = Self::validate_declared_output_keys(node, 1, "union")?;
        let out_key = out_keys.remove(0);

        let first_fmt = Self::resolve_input_format(&inputs[0], input_content_types)?;
        if first_fmt == ColumnarInputFormatV1::OpaqueBinary {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "union requires tabular content_type ('{CONTENT_TYPE_COLUMNAR_JSON_V1}' or '{CONTENT_TYPE_NDJSON}'), got '{CONTENT_TYPE_OCTET_STREAM}' for '{}'",
                    inputs[0].asset_key
                ),
            ));
        }
        for inp in &inputs[1..] {
            let fmt = Self::resolve_input_format(inp, input_content_types)?;
            if fmt != first_fmt {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "union input '{}' content_type '{}' does not match first input format '{}'",
                        inp.asset_key,
                        fmt.content_type(),
                        first_fmt.content_type()
                    ),
                ));
            }
        }

        let (out_bytes, total_rows) = match first_fmt {
            ColumnarInputFormatV1::ColumnarTableV1 => {
                let first_raw = read_verified_cas_input(run_dir, &inputs[0])?;
                let first_table = ColumnarTableV1::from_json_bytes_strict(&first_raw)?;
                let schema = first_table.columns;
                let mut merged_rows = first_table.rows;
                for inp in &inputs[1..] {
                    let raw = read_verified_cas_input(run_dir, inp)?;
                    let tbl = ColumnarTableV1::from_json_bytes_strict(&raw)?;
                    if tbl.columns != schema {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "union columnar schema mismatch across physical inputs",
                        ));
                    }
                    merged_rows.extend(tbl.rows);
                }
                let total_rows = merged_rows.len() as u64;
                let merged = ColumnarTableV1::try_new(schema, merged_rows)?;
                (merged.to_canonical_json_bytes()?, total_rows)
            }
            ColumnarInputFormatV1::Ndjson => {
                let mut merged_lines = Vec::new();
                for inp in inputs {
                    let raw = read_verified_cas_input(run_dir, inp)?;
                    for (trimmed, _) in Self::parse_ndjson_rows(&raw)? {
                        merged_lines.push(trimmed);
                    }
                }
                let total_rows = merged_lines.len() as u64;
                let mut buf = merged_lines.join("\n").into_bytes();
                if !buf.is_empty() {
                    buf.push(b'\n');
                }
                (buf, total_rows)
            }
            ColumnarInputFormatV1::OpaqueBinary => unreachable!(),
        };

        Self::commit_staged_cas_outputs(
            run_dir,
            vec![(out_key, out_bytes, Some(total_rows), first_fmt)],
        )
    }

    /// Passthrough: returns inputs as-is, unchanged.
    fn op_passthrough(inputs: &[AssetInstanceV1]) -> Vec<AssetInstanceV1> {
        inputs.to_vec()
    }

    /// Filter rows: returns inputs with the same fingerprints.
    /// This is metadata-only; actual filtering would happen in a real runner.
    fn op_filter_rows(inputs: &[AssetInstanceV1], _node: &NodeV1) -> Vec<AssetInstanceV1> {
        // In a real runner, this would apply node.params["predicate"] to row data.
        // For metadata-only: we forward asset instances (fingerprints don't change
        // because the op hasn't actually mutated data — DataOpsSession will derive
        // the correct fingerprint during materialization).
        inputs.to_vec()
    }

    /// Union: metadata-only pass-through for union stages.
    /// Actual union output derivation happens during materialization.
    fn op_union(inputs: &[AssetInstanceV1], _node: &NodeV1) -> Vec<AssetInstanceV1> {
        // For metadata-only: return all inputs unchanged. The orchestrator uses
        // materialize_node_outputs to derive canonical output fingerprints.
        inputs.to_vec()
    }
}

impl OpRunner for NativeOpRunner {
    type Error = io::Error;

    fn run<E: RunEventEmitter<Error = Self::Error>>(
        &self,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        emitter: &E,
    ) -> Result<Vec<AssetInstanceV1>, Self::Error> {
        // Without ExecutionContext, we can't derive trace_id or deterministic span_id.
        // This is the trait-level fallback; callers should prefer run_with_context().
        //
        // H-03: derive a per-invocation RunId from the wall-clock so that
        // telemetry from different trait-level calls does not silently collide
        // under a single all-zero sentinel.  The id is NOT cryptographic —
        // it is a best-effort uniqueness measure for the logging layer.
        let clock_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let hash = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b"swarmtorch.trait_fallback_run_id");
            h.update(clock_nanos.to_le_bytes());
            h.finalize()
        };
        let mut run_id_bytes = [0u8; 16];
        run_id_bytes.copy_from_slice(&hash[..16]);
        let ctx = ExecutionContext {
            run_id: RunId::from_bytes(run_id_bytes),
            clock_nanos: || {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
            },
        };
        self.run_with_context(&ctx, node, inputs, emitter)
    }
}

impl NativeOpRunner {
    fn physical_wasm_transform(
        &self,
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        wasm_isolation: Option<WasmIsolationCtx<'_>>,
        input_content_types: &BTreeMap<String, String>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let Some(executor) = self.wasm_executor.as_ref() else {
            // ST-09 containment: untrusted Wasm extension execution fails closed unless an external executor is configured.
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                WasmSandboxError::UnsupportedUntrustedWasmSandbox.to_string(),
            ));
        };
        if node.execution_trust != ExecutionTrust::SandboxedExtension {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "wasm_transform requires ExecutionTrust::SandboxedExtension, got {:?}",
                    node.execution_trust
                ),
            ));
        }
        let Some((policy, registry)) = wasm_isolation else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "wasm_transform requires SandboxedExtension policy and module registry context",
            ));
        };
        if inputs.len() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "wasm_transform requires exactly 1 physical CAS input, got {}",
                    inputs.len()
                ),
            ));
        }
        let mut out_keys = Self::validate_declared_output_keys(node, 1, "wasm_transform")?;
        let out_key = out_keys.remove(0);

        let (_declared_hash, export_name, wasm_bytes) =
            resolve_and_preflight_wasm_node_v1(policy, &node.params, registry)?;

        let fmt = Self::resolve_input_format(&inputs[0], input_content_types)?;
        let raw_input = read_verified_cas_input(run_dir, &inputs[0])?;
        let receipt = executor
            .execute_byte_transform(policy, wasm_bytes, export_name, &raw_input)
            .map_err(wasm_error_to_io)?;

        if (receipt.output_bytes.len() as u64) > MAX_SINGLE_ARTIFACT_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "wasm_transform output ({} bytes) exceeds MAX_SINGLE_ARTIFACT_BYTES ({MAX_SINGLE_ARTIFACT_BYTES})",
                    receipt.output_bytes.len()
                ),
            ));
        }
        let rows_opt = match fmt {
            ColumnarInputFormatV1::ColumnarTableV1 => {
                Some(ColumnarTableV1::from_json_bytes_strict(&receipt.output_bytes)?.row_count)
            }
            ColumnarInputFormatV1::Ndjson => {
                Some(Self::parse_ndjson_rows(&receipt.output_bytes)?.len() as u64)
            }
            ColumnarInputFormatV1::OpaqueBinary => None,
        };

        Self::commit_staged_cas_outputs(
            run_dir,
            vec![(out_key, receipt.output_bytes, rows_opt, fmt)],
        )
    }
}

fn write_cas_bytes(run_dir: &Path, bytes: &[u8]) -> io::Result<(String, String, u64)> {
    let digest = Sha256::digest(bytes);
    let mut sha = [0u8; 32];
    sha.copy_from_slice(&digest[..]);
    let _ = write_once_cas_blob(run_dir, &sha, bytes)?;
    let hex = hex_lower(&sha);
    let uri = format!("cas://sha256/{hex}");
    Ok((hex, uri, bytes.len() as u64))
}

fn read_verified_cas_input(run_dir: &Path, instance: &AssetInstanceV1) -> io::Result<Vec<u8>> {
    let uri = instance.uri.as_deref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("input '{}' has no physical CAS URI", instance.asset_key),
        )
    })?;
    let expected_sha = parse_cas_uri_digest(uri).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "input '{}' URI '{uri}' is not a valid cas://sha256 URI",
                instance.asset_key
            ),
        )
    })?;
    let blob_path = run_dir
        .join("blobs")
        .join("sha256")
        .join(hex_lower(&expected_sha));
    let bytes = read_bytes_bounded(&blob_path, MAX_SINGLE_ARTIFACT_BYTES)?;
    let actual_digest = Sha256::digest(&bytes);
    if actual_digest.as_slice() != expected_sha {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "CAS blob SHA-256 verification failed for '{}'",
                instance.asset_key
            ),
        ));
    }
    Ok(bytes)
}

fn canon_param_str<'a>(params: &'a CanonParams, key: &str) -> Option<&'a str> {
    match params.get(key) {
        Some(CanonValue::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn cell_matches_predicate(cell: &Value, op: &str, rhs_raw: &str) -> bool {
    if cell.is_null() {
        return match op {
            "eq" => rhs_raw == "null",
            "ne" => rhs_raw != "null",
            _ => false,
        };
    }
    if let Some(b) = cell.as_bool() {
        if let Ok(rhs_b) = rhs_raw.parse::<bool>() {
            return match op {
                "eq" => b == rhs_b,
                "ne" => b != rhs_b,
                _ => false,
            };
        }
        return false;
    }
    if let Value::Number(lhs_num) = cell {
        let Some(rhs_num) = parse_finite_json_number_str(rhs_raw) else {
            return false;
        };
        let Some(ord) = cmp_json_numbers(lhs_num, &rhs_num) else {
            return false;
        };
        return match op {
            "eq" => ord == std::cmp::Ordering::Equal,
            "ne" => ord != std::cmp::Ordering::Equal,
            "gt" => ord == std::cmp::Ordering::Greater,
            "ge" => matches!(ord, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
            "lt" => ord == std::cmp::Ordering::Less,
            "le" => matches!(ord, std::cmp::Ordering::Less | std::cmp::Ordering::Equal),
            _ => false,
        };
    }
    if let Some(lhs_s) = cell.as_str() {
        return match op {
            "eq" => lhs_s == rhs_raw,
            "ne" => lhs_s != rhs_raw,
            "gt" => lhs_s > rhs_raw,
            "ge" => lhs_s >= rhs_raw,
            "lt" => lhs_s < rhs_raw,
            "le" => lhs_s <= rhs_raw,
            _ => false,
        };
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::observe::{EventRecord, MetricRecord};
    use swarm_torch_core::run_graph::{AssetRefV1, CanonParams, ExecutionTrust, NodeV1, OpKind};

    /// Test emitter that captures spans.
    struct TestEmitter {
        spans: std::sync::RwLock<Vec<SpanRecord>>,
    }

    impl TestEmitter {
        fn new() -> Self {
            Self {
                spans: std::sync::RwLock::new(Vec::new()),
            }
        }
    }

    impl RunEventEmitter for TestEmitter {
        type Error = io::Error;

        fn emit_span(&self, span: &SpanRecord) -> io::Result<()> {
            self.spans.write().unwrap().push(span.clone());
            Ok(())
        }

        fn emit_event(&self, _event: &EventRecord) -> io::Result<()> {
            Ok(())
        }

        fn emit_metric(&self, _metric: &MetricRecord) -> io::Result<()> {
            Ok(())
        }
    }

    fn test_ctx() -> ExecutionContext {
        ExecutionContext {
            run_id: RunId::from_bytes([42u8; 16]),
            clock_nanos: {
                static COUNTER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(1_000_000_000);
                || COUNTER.fetch_add(1_000_000, std::sync::atomic::Ordering::SeqCst)
            },
        }
    }

    fn test_node(op_type: &str) -> NodeV1 {
        NodeV1 {
            node_key: "test/node".to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: op_type.to_string(),
            inputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/raw".to_string(),
                fingerprint: None,
            }],
            outputs: vec![],
            params: CanonParams::new(),
            code_ref: Some("test@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
            execution_hint: None,
            cache_policy: None,
            materialization_policy: None,
            resources: None,
            op_hash: None,
        }
    }

    fn test_inputs() -> Vec<AssetInstanceV1> {
        vec![AssetInstanceV1 {
            asset_key: "dataset://ns/raw".to_string(),
            fingerprint_v0: "a".repeat(64),
            uri: Some("s3://bucket/raw".to_string()),
        }]
    }

    #[test]
    fn passthrough_preserves_inputs() {
        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests);
        let node = test_node("passthrough");
        let inputs = test_inputs();

        let outputs = runner
            .run_with_context(&ctx, &node, &inputs, &emitter)
            .unwrap();

        // Outputs should be identical to inputs
        assert_eq!(outputs.len(), inputs.len());
        assert_eq!(outputs[0].asset_key, inputs[0].asset_key);
        assert_eq!(outputs[0].fingerprint_v0, inputs[0].fingerprint_v0);
        assert_eq!(outputs[0].uri, inputs[0].uri);

        // Should have emitted exactly one span
        let spans = emitter.spans.read().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "op/passthrough");
        assert!(spans[0].end_unix_nanos.is_some());

        // trace_id should be derived from run_id
        assert_eq!(spans[0].trace_id, TraceId::from_bytes([42u8; 16]));
    }

    #[test]
    fn filter_rows_metadata_only() {
        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests);
        let node = test_node("filter_rows");
        let inputs = test_inputs();

        let outputs = runner
            .run_with_context(&ctx, &node, &inputs, &emitter)
            .unwrap();

        // Metadata-only: outputs = inputs (actual filtering deferred to real runner)
        assert_eq!(outputs.len(), inputs.len());
        assert_eq!(outputs[0].asset_key, inputs[0].asset_key);

        // Span emitted
        let spans = emitter.spans.read().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "op/filter_rows");
    }

    #[test]
    fn union_metadata_only_forwards_inputs() {
        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests);
        let node = test_node("union");
        let inputs = test_inputs();

        let outputs = runner
            .run_with_context(&ctx, &node, &inputs, &emitter)
            .unwrap();

        assert_eq!(outputs, inputs);

        let spans = emitter.spans.read().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "op/union");
    }

    #[test]
    fn deterministic_span_id_is_stable() {
        // Same (node_id, ts) → same span_id
        let node_a = [42u8; 16];
        let node_b = [43u8; 16];
        let id1 = deterministic_span_id(&node_a, 1_000_000_000);
        let id2 = deterministic_span_id(&node_a, 1_000_000_000);
        assert_eq!(id1, id2);

        // Different ts → different span_id
        let id3 = deterministic_span_id(&node_a, 2_000_000_000);
        assert_ne!(id1, id3);

        // Different node_id → different span_id
        let id4 = deterministic_span_id(&node_b, 1_000_000_000);
        assert_ne!(id1, id4);
    }

    #[test]
    fn unsupported_op_type_returns_error() {
        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests);
        let node = test_node("nonexistent_op");
        let inputs = test_inputs();

        let result = runner.run_with_context(&ctx, &node, &inputs, &emitter);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unsupported op_type"));
    }

    #[test]
    fn native_runner_spans_use_swarmtorch_namespace() {
        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests);
        let node = test_node("passthrough");
        let inputs = test_inputs();

        let _ = runner
            .run_with_context(&ctx, &node, &inputs, &emitter)
            .expect("run should succeed");
        let spans = emitter.spans.read().unwrap();
        assert_eq!(spans.len(), 1);
        let attrs = &spans[0].attrs;

        assert!(attrs.contains_key("swarmtorch.op_type"));
        assert!(attrs.contains_key("swarmtorch.node_key"));
        assert!(attrs.contains_key("swarmtorch.input_count"));
        assert!(attrs.contains_key("swarmtorch.output_count"));
        assert!(!attrs.contains_key("op_type"));
        assert!(!attrs.contains_key("node_key"));
        assert!(!attrs.contains_key("input_count"));
        assert!(!attrs.contains_key("output_count"));
    }

    #[test]
    fn wasm_transform_fails_closed_when_executor_unattached() {
        let pid = std::process::id();
        let run_dir = std::env::temp_dir().join(format!("swarmtorch_native_wasm_deny_{pid}"));
        let _ = std::fs::remove_dir_all(&run_dir);
        std::fs::create_dir_all(&run_dir).unwrap();

        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::new();
        let node = test_node("wasm_transform");
        let policy = SandboxedExtensionPolicyV1::default();
        let registry = BTreeMap::new();

        let err = runner
            .run_in_session(
                &ctx,
                &run_dir,
                &node,
                &[],
                &emitter,
                Some((&policy, &registry)),
            )
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        assert!(err
            .to_string()
            .contains(&WasmSandboxError::UnsupportedUntrustedWasmSandbox.to_string()));
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn physical_columnar_filter_and_union_enforce_schema_and_write_cas_blobs() {
        use swarm_torch_core::observe::RunId;

        let pid = std::process::id();
        let run_dir = std::env::temp_dir().join(format!("swarmtorch_native_col_{pid}"));
        let _ = std::fs::remove_dir_all(&run_dir);
        std::fs::create_dir_all(&run_dir).unwrap();

        let cols = vec![
            ColumnSchemaV1 {
                name: "id".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            },
            ColumnSchemaV1 {
                name: "score".to_string(),
                dtype: ColumnDtypeV1::F64,
                nullable: false,
            },
        ];
        let tbl1 = ColumnarTableV1::try_new(
            cols.clone(),
            vec![
                vec![Value::from(1), Value::from(0.9)],
                vec![Value::from(2), Value::from(0.2)],
                vec![Value::from(3), Value::from(0.8)],
            ],
        )
        .unwrap();
        let (hex1, uri1, _) =
            write_cas_bytes(&run_dir, &tbl1.to_canonical_json_bytes().unwrap()).unwrap();
        let in1 = AssetInstanceV1 {
            asset_key: "dataset://ns/shard1".to_string(),
            fingerprint_v0: hex1,
            uri: Some(uri1),
        };

        let mut filter_node = test_node("filter_rows");
        filter_node
            .params
            .insert("column".to_string(), CanonValue::Str("score".to_string()));
        filter_node
            .params
            .insert("op".to_string(), CanonValue::Str("ge".to_string()));
        filter_node
            .params
            .insert("value".to_string(), CanonValue::Str("0.8".to_string()));
        filter_node.outputs = vec![AssetRefV1 {
            asset_key: "dataset://ns/filtered".to_string(),
            fingerprint: None,
        }];

        let runner = NativeOpRunner::new();
        let ctx = ExecutionContext {
            run_id: RunId::from_bytes([7; 16]),
            clock_nanos: || 1_000_000,
        };
        let emitter = TestEmitter::new();
        let mut cts = BTreeMap::new();
        cts.insert(
            "dataset://ns/shard1".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );

        let filter_out = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &filter_node,
                std::slice::from_ref(&in1),
                &emitter,
                None,
                &cts,
            )
            .unwrap();
        assert_eq!(filter_out.output_specs[0].rows, Some(2));
        assert!(filter_out.output_specs[0].bytes.unwrap() > 0);
        assert_eq!(
            filter_out
                .output_content_types
                .get("dataset://ns/filtered")
                .map(String::as_str),
            Some(CONTENT_TYPE_COLUMNAR_JSON_V1)
        );
        let kept_raw = read_verified_cas_input(&run_dir, &filter_out.instances[0]).unwrap();
        let kept_tbl = ColumnarTableV1::from_json_bytes_strict(&kept_raw).unwrap();
        assert_eq!(kept_tbl.row_count, 2);

        // Compatible union succeeds and merges rows
        let tbl2 =
            ColumnarTableV1::try_new(cols, vec![vec![Value::from(4), Value::from(0.95)]]).unwrap();
        let (hex2, uri2, _) =
            write_cas_bytes(&run_dir, &tbl2.to_canonical_json_bytes().unwrap()).unwrap();
        let in2 = AssetInstanceV1 {
            asset_key: "dataset://ns/shard2".to_string(),
            fingerprint_v0: hex2,
            uri: Some(uri2),
        };
        cts.extend(filter_out.output_content_types.clone());
        cts.insert(
            "dataset://ns/shard2".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        let mut union_node = test_node("union");
        union_node.outputs = vec![AssetRefV1 {
            asset_key: "dataset://ns/merged".to_string(),
            fingerprint: None,
        }];
        let union_out = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &union_node,
                &[filter_out.instances[0].clone(), in2],
                &emitter,
                None,
                &cts,
            )
            .unwrap();
        assert_eq!(union_out.output_specs[0].rows, Some(3));
        assert_eq!(
            union_out
                .output_content_types
                .get("dataset://ns/merged")
                .map(String::as_str),
            Some(CONTENT_TYPE_COLUMNAR_JSON_V1)
        );

        // Incompatible schema union fails closed with InvalidData
        let bad_cols = vec![ColumnSchemaV1 {
            name: "different_col".to_string(),
            dtype: ColumnDtypeV1::Utf8,
            nullable: false,
        }];
        let bad_tbl = ColumnarTableV1::try_new(bad_cols, vec![vec![Value::from("x")]]).unwrap();
        let (hex3, uri3, _) =
            write_cas_bytes(&run_dir, &bad_tbl.to_canonical_json_bytes().unwrap()).unwrap();
        let in_bad = AssetInstanceV1 {
            asset_key: "dataset://ns/bad".to_string(),
            fingerprint_v0: hex3,
            uri: Some(uri3),
        };
        cts.insert(
            "dataset://ns/bad".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        let err = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &union_node,
                &[in1, in_bad],
                &emitter,
                None,
                &cts,
            )
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn st10_physical_strict_rejects_missing_or_non_cas_inputs_and_run_with_context() {
        let pid = std::process::id();
        let run_dir = std::env::temp_dir().join(format!("swarmtorch_native_strict_{pid}"));
        let _ = std::fs::remove_dir_all(&run_dir);
        std::fs::create_dir_all(&run_dir).unwrap();

        let runner = NativeOpRunner::new();
        assert_eq!(runner.mode(), RunnerExecutionModeV1::PhysicalStrict);
        assert_eq!(
            NativeOpRunner::default().mode(),
            RunnerExecutionModeV1::PhysicalStrict
        );

        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let node = test_node("passthrough");
        let non_cas_inputs = test_inputs();

        // Direct OpRunner metadata-only execution rejects PhysicalStrict
        let err = runner
            .run_with_context(&ctx, &node, &non_cas_inputs, &emitter)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);

        // Session execution rejects empty inputs and s3:// URIs rather than falling back to metadata-only
        let err_empty = runner
            .run_in_session(&ctx, &run_dir, &node, &[], &emitter, None)
            .unwrap_err();
        assert_eq!(err_empty.kind(), io::ErrorKind::InvalidInput);

        let err_s3 = runner
            .run_in_session(&ctx, &run_dir, &node, &non_cas_inputs, &emitter, None)
            .unwrap_err();
        assert_eq!(err_s3.kind(), io::ErrorKind::InvalidInput);

        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn st10_content_type_dispatch_rejects_sniffing_and_parses_ndjson_starting_with_brace() {
        let pid = std::process::id();
        let run_dir = std::env::temp_dir().join(format!("swarmtorch_native_ct_{pid}"));
        let _ = std::fs::remove_dir_all(&run_dir);
        std::fs::create_dir_all(&run_dir).unwrap();

        // Multi-row NDJSON starts with `{`, which previously triggered false ColumnarTableV1 sniffing!
        let ndjson_bytes = br#"{"id":1,"score":0.9}
{"id":2,"score":0.2}
{"id":3,"score":0.85}
"#;
        let (hex_nd, uri_nd, _) = write_cas_bytes(&run_dir, ndjson_bytes).unwrap();
        let inp_nd = AssetInstanceV1 {
            asset_key: "dataset://ns/nd".to_string(),
            fingerprint_v0: hex_nd,
            uri: Some(uri_nd),
        };
        let mut filter_node = test_node("filter_rows");
        filter_node
            .params
            .insert("column".to_string(), CanonValue::Str("score".to_string()));
        filter_node
            .params
            .insert("op".to_string(), CanonValue::Str("ge".to_string()));
        filter_node
            .params
            .insert("value".to_string(), CanonValue::Str("0.8".to_string()));
        filter_node.outputs = vec![AssetRefV1 {
            asset_key: "dataset://ns/nd_filtered".to_string(),
            fingerprint: None,
        }];

        let runner = NativeOpRunner::new();
        let ctx = test_ctx();
        let emitter = TestEmitter::new();

        // 1. Missing declared content_type fails closed (no `{` sniffing fallback)
        let err_missing_ct = runner
            .run_in_session(
                &ctx,
                &run_dir,
                &filter_node,
                std::slice::from_ref(&inp_nd),
                &emitter,
                None,
            )
            .unwrap_err();
        assert_eq!(err_missing_ct.kind(), io::ErrorKind::InvalidInput);

        // 2. Wrong declared content_type (e.g. application/octet-stream or mismatched ColumnarTableV1) fails closed
        let mut wrong_cts = BTreeMap::new();
        wrong_cts.insert(
            "dataset://ns/nd".to_string(),
            CONTENT_TYPE_OCTET_STREAM.to_string(),
        );
        let err_wrong_ct = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &filter_node,
                std::slice::from_ref(&inp_nd),
                &emitter,
                None,
                &wrong_cts,
            )
            .unwrap_err();
        assert_eq!(err_wrong_ct.kind(), io::ErrorKind::InvalidInput);

        wrong_cts.insert(
            "dataset://ns/nd".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        let err_mislabeled = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &filter_node,
                std::slice::from_ref(&inp_nd),
                &emitter,
                None,
                &wrong_cts,
            )
            .unwrap_err();
        assert_eq!(err_mislabeled.kind(), io::ErrorKind::InvalidData);

        // 3. Declared NDJSON content_type parses cleanly even though payload starts with `{`
        let mut nd_cts = BTreeMap::new();
        nd_cts.insert(
            "dataset://ns/nd".to_string(),
            CONTENT_TYPE_NDJSON.to_string(),
        );
        let out = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &filter_node,
                std::slice::from_ref(&inp_nd),
                &emitter,
                None,
                &nd_cts,
            )
            .unwrap();
        assert_eq!(out.output_specs[0].rows, Some(2));
        assert_eq!(
            out.output_content_types
                .get("dataset://ns/nd_filtered")
                .map(String::as_str),
            Some(CONTENT_TYPE_NDJSON)
        );

        // 4. Partial filter_rows params (`column` without `value`) and unknown `op` fail closed (`AR-14-08`)!
        let mut partial_node = filter_node.clone();
        partial_node.params.remove("value");
        let err_partial = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &partial_node,
                std::slice::from_ref(&inp_nd),
                &emitter,
                None,
                &nd_cts,
            )
            .unwrap_err();
        assert_eq!(err_partial.kind(), io::ErrorKind::InvalidInput);

        let mut bad_op_node = filter_node;
        bad_op_node
            .params
            .insert("op".to_string(), CanonValue::Str("regex_match".to_string()));
        let err_op = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &bad_op_node,
                std::slice::from_ref(&inp_nd),
                &emitter,
                None,
                &nd_cts,
            )
            .unwrap_err();
        assert_eq!(err_op.kind(), io::ErrorKind::InvalidInput);

        // 5. Oversized NDJSON line (> MAX_NDJSON_LINE_BYTES) fails closed before serde_json allocation!
        let oversized_line = vec![b'a'; MAX_NDJSON_LINE_BYTES + 1];
        let err_oversized = NativeOpRunner::parse_ndjson_rows(&oversized_line).unwrap_err();
        assert_eq!(err_oversized.kind(), io::ErrorKind::InvalidData);

        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn st15_columnar_bounds_duplicate_columns_digest_and_i64_predicates() {
        use crate::wasm_sandbox::{
            SandboxedExtensionPolicyV1, WasmExecutionReceiptV1, WasmExtensionExecutor,
            WASM_MAGIC_AND_VERSION_V1,
        };

        // 1. Duplicate column names rejected cleanly.
        let dup_cols = vec![
            ColumnSchemaV1 {
                name: "x".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            },
            ColumnSchemaV1 {
                name: "x".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            },
        ];
        assert_eq!(
            ColumnarTableV1::try_new(dup_cols, vec![vec![Value::from(1), Value::from(2)]])
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );

        // 2. Column name > 256 chars and rows.len() > MAX_COLUMNAR_ROWS rejected.
        let long_name = vec![ColumnSchemaV1 {
            name: "a".repeat(257),
            dtype: ColumnDtypeV1::I64,
            nullable: false,
        }];
        assert_eq!(
            ColumnarTableV1::try_new(long_name, vec![])
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let too_many = ColumnarTableV1 {
            schema_version: 1,
            table_format: "stcl-json-v1".to_string(),
            row_count: (MAX_COLUMNAR_ROWS + 1) as u64,
            columns: vec![ColumnSchemaV1 {
                name: "id".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            }],
            rows: vec![vec![Value::from(1)]; MAX_COLUMNAR_ROWS + 1],
            content_digest_sha256: None,
        };
        assert_eq!(
            too_many.validate().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        // 3. content_digest_sha256 verified against (columns, rows); cell tamper fails from_json_bytes_strict!
        let mut valid_tbl = ColumnarTableV1::try_new(
            vec![ColumnSchemaV1 {
                name: "id".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            }],
            vec![
                vec![Value::from(9_007_199_254_740_992_i64)],
                vec![Value::from(9_007_199_254_740_993_i64)],
            ],
        )
        .unwrap();
        let valid_bytes = valid_tbl.to_canonical_json_bytes().unwrap();
        assert!(ColumnarTableV1::from_json_bytes_strict(&valid_bytes).is_ok());

        valid_tbl.rows[0][0] = Value::from(9_007_199_254_740_994_i64);
        let tampered_bytes = serde_json::to_vec(&valid_tbl).unwrap();
        assert_eq!(
            ColumnarTableV1::from_json_bytes_strict(&tampered_bytes)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );

        // 4. Exact > 2^53 i64 / u64 / boundary float comparisons in cell_matches_predicate!
        let big_even = Value::from(9_007_199_254_740_992_i64);
        let big_odd = Value::from(9_007_199_254_740_993_i64);
        assert!(cell_matches_predicate(&big_even, "eq", "9007199254740992"));
        assert!(!cell_matches_predicate(&big_odd, "eq", "9007199254740992"));
        assert!(cell_matches_predicate(&big_odd, "ne", "9007199254740992"));
        assert!(cell_matches_predicate(&big_odd, "gt", "9007199254740992"));
        assert!(!cell_matches_predicate(
            &big_odd,
            "eq",
            "9007199254740992.0"
        ));
        assert!(cell_matches_predicate(&big_odd, "gt", "9007199254740992.0"));
        assert!(cell_matches_predicate(
            &Value::from(u64::MAX),
            "gt",
            "18446744073709551614"
        ));
        assert!(!cell_matches_predicate(
            &Value::from(u64::MAX),
            "eq",
            "18446744073709551614"
        ));
        assert!(cell_matches_predicate(
            &Value::from(i64::MAX),
            "lt",
            "9223372036854775808.0"
        ));
        assert!(cell_matches_predicate(
            &Value::from(i64::MIN),
            "gt",
            "-9223372036854779904.0"
        ));
        assert!(cell_matches_predicate(
            &Value::from(i64::MIN),
            "eq",
            "-9223372036854775808.0"
        ));
        assert!(!cell_matches_predicate(&big_odd, "eq", "NaN"));
        assert!(!cell_matches_predicate(&big_odd, "gt", "inf"));

        // 5. Output count enforcement before CAS writes across passthrough, filter_rows, union, wasm_transform!
        let pid = std::process::id();
        let run_dir = std::env::temp_dir().join(format!("swarmtorch_native_out_contract_{pid}"));
        let _ = std::fs::remove_dir_all(&run_dir);
        std::fs::create_dir_all(&run_dir).unwrap();

        let (hex_in, uri_in, _) = write_cas_bytes(&run_dir, &valid_bytes).unwrap();
        let inp = AssetInstanceV1 {
            asset_key: "dataset://ns/raw".to_string(),
            fingerprint_v0: hex_in.clone(),
            uri: Some(uri_in),
        };
        let mut cts = BTreeMap::new();
        cts.insert(
            "dataset://ns/raw".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::new();

        for op in ["passthrough", "filter_rows", "union"] {
            let mut node = test_node(op);
            node.outputs = vec![];
            let err_empty = runner
                .run_in_session_with_content_types(
                    &ctx,
                    &run_dir,
                    &node,
                    std::slice::from_ref(&inp),
                    &emitter,
                    None,
                    &cts,
                )
                .unwrap_err();
            assert_eq!(err_empty.kind(), io::ErrorKind::InvalidInput);

            node.outputs = vec![
                AssetRefV1 {
                    asset_key: "dataset://ns/o1".to_string(),
                    fingerprint: None,
                },
                AssetRefV1 {
                    asset_key: "dataset://ns/o2".to_string(),
                    fingerprint: None,
                },
            ];
            let err_two = runner
                .run_in_session_with_content_types(
                    &ctx,
                    &run_dir,
                    &node,
                    std::slice::from_ref(&inp),
                    &emitter,
                    None,
                    &cts,
                )
                .unwrap_err();
            assert_eq!(err_two.kind(), io::ErrorKind::InvalidInput);
        }

        // Exact filter_rows execution with 1 declared output filters 2^53+1 cleanly!
        let mut filter_node = test_node("filter_rows");
        filter_node
            .params
            .insert("column".to_string(), CanonValue::Str("id".to_string()));
        filter_node
            .params
            .insert("op".to_string(), CanonValue::Str("eq".to_string()));
        filter_node.params.insert(
            "value".to_string(),
            CanonValue::Str("9007199254740993".to_string()),
        );
        filter_node.outputs = vec![AssetRefV1 {
            asset_key: "dataset://ns/filtered".to_string(),
            fingerprint: None,
        }];
        let out = runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &filter_node,
                std::slice::from_ref(&inp),
                &emitter,
                None,
                &cts,
            )
            .unwrap();
        let kept = ColumnarTableV1::from_json_bytes_strict(
            &read_verified_cas_input(&run_dir, &out.instances[0]).unwrap(),
        )
        .unwrap();
        assert_eq!(kept.rows.len(), 1);
        assert_eq!(kept.rows[0][0].as_i64(), Some(9_007_199_254_740_993_i64));

        // wasm_transform rejects 0 declared outputs up front (before writing any CAS output)!
        struct EchoWasm;
        impl WasmExtensionExecutor for EchoWasm {
            fn execute_byte_transform(
                &self,
                policy: &SandboxedExtensionPolicyV1,
                wasm_bytes: &[u8],
                _export_name: &str,
                input_payload: &[u8],
            ) -> Result<WasmExecutionReceiptV1, WasmSandboxError> {
                Ok(WasmExecutionReceiptV1 {
                    module_sha256: SandboxedExtensionPolicyV1::sha256_module(wasm_bytes),
                    fuel_consumed: policy.limits.max_fuel.min(10),
                    peak_memory_bytes: 65_536,
                    return_code: 0,
                    output_bytes: input_payload.to_vec(),
                })
            }
            fn execute_scalar_i32(
                &self,
                _policy: &SandboxedExtensionPolicyV1,
                _wasm_bytes: &[u8],
                _export_name: &str,
                _arg: i32,
            ) -> Result<(i32, WasmExecutionReceiptV1), WasmSandboxError> {
                unreachable!()
            }
        }
        let mut wasm = WASM_MAGIC_AND_VERSION_V1.to_vec();
        wasm.extend_from_slice(&[1, 7, 1, 0x60, 2, 0x7f, 0x7f, 1, 0x7f]);
        wasm.extend_from_slice(&[3, 2, 1, 0]);
        wasm.extend_from_slice(&[5, 4, 1, 0x01, 1, 1]);
        wasm.extend_from_slice(&[7, 13, 1, 9]);
        wasm.extend_from_slice(b"transform");
        wasm.extend_from_slice(&[0x00, 0x00]);
        wasm.extend_from_slice(&[10, 6, 1, 4, 0, 0x41, 0, 0x0b]);
        let wasm_hash = SandboxedExtensionPolicyV1::sha256_module(&wasm);
        let wasm_policy = SandboxedExtensionPolicyV1::with_allowed_hash(wasm_hash);
        let mut wasm_reg = BTreeMap::new();
        wasm_reg.insert(wasm_hash, wasm);
        let wasm_runner = NativeOpRunner::new().with_wasm_executor(Arc::new(EchoWasm));
        let mut wasm_node = test_node("wasm_transform");
        wasm_node.execution_trust = ExecutionTrust::SandboxedExtension;
        wasm_node.params.insert(
            "module_sha256".to_string(),
            CanonValue::Str(hex_lower(&wasm_hash)),
        );
        wasm_node.params.insert(
            "export".to_string(),
            CanonValue::Str("transform".to_string()),
        );
        wasm_node.outputs = vec![];
        let err_wasm_zero_out = wasm_runner
            .run_in_session_with_content_types(
                &ctx,
                &run_dir,
                &wasm_node,
                std::slice::from_ref(&inp),
                &emitter,
                Some((&wasm_policy, &wasm_reg)),
                &cts,
            )
            .unwrap_err();
        assert_eq!(err_wasm_zero_out.kind(), io::ErrorKind::InvalidInput);
        let _ = hex_in;
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn st15_multi_input_duplicate_outputs_and_partial_failure_write_zero_orphan_cas_blobs() {
        use crate::artifacts::{DataOpsSession, RunArtifactBundle, RunArtifactSink};
        use crate::scheduler::execute_graph_sequential;
        use swarm_torch_core::dataops::{AuthModeMarker, SourceDescriptorV0, TrustClass};
        use swarm_torch_core::execution::PermissivePolicy;
        use swarm_torch_core::run_graph::GraphV1;

        let pid = std::process::id();
        let base = std::env::temp_dir().join(format!("swarmtorch_native_no_orphans_{pid}"));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();

        let tbl_a = ColumnarTableV1::try_new(
            vec![ColumnSchemaV1 {
                name: "k".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            }],
            vec![vec![Value::from(1)]],
        )
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();
        let tbl_b = ColumnarTableV1::try_new(
            vec![ColumnSchemaV1 {
                name: "k".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            }],
            vec![vec![Value::from(2)]],
        )
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();

        let (hex_a, uri_a, _) = write_cas_bytes(&base, &tbl_a).unwrap();
        let (hex_b, uri_b, _) = write_cas_bytes(&base, &tbl_b).unwrap();
        let in_a = AssetInstanceV1 {
            asset_key: "dataset://ns/a".to_string(),
            fingerprint_v0: hex_a.clone(),
            uri: Some(uri_a.clone()),
        };
        let in_b = AssetInstanceV1 {
            asset_key: "dataset://ns/b".to_string(),
            fingerprint_v0: hex_b.clone(),
            uri: Some(uri_b.clone()),
        };
        let mut cts = BTreeMap::new();
        cts.insert(
            "dataset://ns/a".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        cts.insert(
            "dataset://ns/b".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        let blob_count = |dir: &Path| -> usize {
            std::fs::read_dir(dir.join("blobs").join("sha256"))
                .map(|it| it.count())
                .unwrap_or(0)
        };

        let ctx = test_ctx();
        let emitter = TestEmitter::new();
        let runner = NativeOpRunner::new();

        // 1. Duplicate and empty declared node.outputs rejected for passthrough & filter_rows before writing CAS!
        for op in ["passthrough", "filter_rows"] {
            let mut dup_node = test_node(op);
            dup_node.inputs = vec![
                AssetRefV1 {
                    asset_key: "dataset://ns/a".to_string(),
                    fingerprint: None,
                },
                AssetRefV1 {
                    asset_key: "dataset://ns/b".to_string(),
                    fingerprint: None,
                },
            ];
            dup_node.outputs = vec![
                AssetRefV1 {
                    asset_key: "dataset://ns/dup".to_string(),
                    fingerprint: None,
                },
                AssetRefV1 {
                    asset_key: "dataset://ns/dup".to_string(),
                    fingerprint: None,
                },
            ];
            // Write distinct content under a custom payload so output digests would differ from input if written:
            let tbl_c = ColumnarTableV1::try_new(
                vec![ColumnSchemaV1 {
                    name: "k".to_string(),
                    dtype: ColumnDtypeV1::I64,
                    nullable: false,
                }],
                vec![vec![Value::from(1)], vec![Value::from(2)]],
            )
            .unwrap()
            .to_canonical_json_bytes()
            .unwrap();
            let (hex_c, uri_c, _) = write_cas_bytes(&base, &tbl_c).unwrap();
            let in_c = AssetInstanceV1 {
                asset_key: "dataset://ns/b".to_string(),
                fingerprint_v0: hex_c,
                uri: Some(uri_c),
            };
            let before = blob_count(&base);
            if op == "filter_rows" {
                dup_node
                    .params
                    .insert("column".to_string(), CanonValue::Str("k".to_string()));
                dup_node
                    .params
                    .insert("op".to_string(), CanonValue::Str("eq".to_string()));
                dup_node
                    .params
                    .insert("value".to_string(), CanonValue::Str("1".to_string()));
            }
            let err_dup = runner
                .run_in_session_with_content_types(
                    &ctx,
                    &base,
                    &dup_node,
                    &[in_a.clone(), in_c.clone()],
                    &emitter,
                    None,
                    &cts,
                )
                .unwrap_err();
            assert_eq!(err_dup.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(blob_count(&base), before, "{op} dup leaked CAS blobs");

            let mut empty_out_node = dup_node.clone();
            empty_out_node.outputs[1].asset_key.clear();
            let err_empty = runner
                .run_in_session_with_content_types(
                    &ctx,
                    &base,
                    &empty_out_node,
                    &[in_a.clone(), in_c],
                    &emitter,
                    None,
                    &cts,
                )
                .unwrap_err();
            assert_eq!(err_empty.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(blob_count(&base), before, "{op} empty key leaked CAS blobs");
        }

        // 2. Partial failure on inputs[1] (`filter_rows` missing column on second input,
        //    `passthrough` missing content_type on second input) writes ZERO CAS blobs for inputs[0]!
        let bad_filter_ndjson = br#"{"other":99}
"#;
        let (hex_nd, uri_nd, _) = write_cas_bytes(&base, bad_filter_ndjson).unwrap();
        let in_nd = AssetInstanceV1 {
            asset_key: "dataset://ns/bad_nd".to_string(),
            fingerprint_v0: hex_nd,
            uri: Some(uri_nd),
        };
        let mut cts_partial = cts.clone();
        cts_partial.insert(
            "dataset://ns/bad_nd".to_string(),
            CONTENT_TYPE_NDJSON.to_string(),
        );
        let mut filter_partial = test_node("filter_rows");
        filter_partial.inputs = vec![
            AssetRefV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint: None,
            },
            AssetRefV1 {
                asset_key: "dataset://ns/bad_nd".to_string(),
                fingerprint: None,
            },
        ];
        filter_partial.outputs = vec![
            AssetRefV1 {
                asset_key: "dataset://ns/out_a".to_string(),
                fingerprint: None,
            },
            AssetRefV1 {
                asset_key: "dataset://ns/out_b".to_string(),
                fingerprint: None,
            },
        ];
        // Use missing-column ColumnarTableV1 on inputs[1] so filter_rows fails on second input
        let missing_col_tbl = ColumnarTableV1::try_new(
            vec![ColumnSchemaV1 {
                name: "other".to_string(),
                dtype: ColumnDtypeV1::I64,
                nullable: false,
            }],
            vec![vec![Value::from(9)]],
        )
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();
        let (hex_mcol, uri_mcol, _) = write_cas_bytes(&base, &missing_col_tbl).unwrap();
        let in_mcol = AssetInstanceV1 {
            asset_key: "dataset://ns/mcol".to_string(),
            fingerprint_v0: hex_mcol,
            uri: Some(uri_mcol),
        };
        cts_partial.insert(
            "dataset://ns/mcol".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        filter_partial
            .params
            .insert("column".to_string(), CanonValue::Str("k".to_string()));
        filter_partial
            .params
            .insert("op".to_string(), CanonValue::Str("eq".to_string()));
        filter_partial
            .params
            .insert("value".to_string(), CanonValue::Str("999".to_string()));
        // Note: input[0] would produce filtered (0 rows -> NEW CAS digest!), input[1] fails column lookup -> zero blobs!
        let before_filter = blob_count(&base);
        let err_partial_filter = runner
            .run_in_session_with_content_types(
                &ctx,
                &base,
                &filter_partial,
                &[in_a.clone(), in_mcol],
                &emitter,
                None,
                &cts_partial,
            )
            .unwrap_err();
        assert_eq!(err_partial_filter.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(blob_count(&base), before_filter);

        // passthrough with unknown/corrupt second input writes zero blobs
        let mut pt_partial = test_node("passthrough");
        pt_partial.outputs = filter_partial.outputs.clone();
        let mut cts_missing_b = BTreeMap::new();
        cts_missing_b.insert(
            "dataset://ns/a".to_string(),
            CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
        );
        let before_pt = blob_count(&base);
        let err_pt = runner
            .run_in_session_with_content_types(
                &ctx,
                &base,
                &pt_partial,
                &[in_a, in_nd],
                &emitter,
                None,
                &cts_missing_b,
            )
            .unwrap_err();
        assert_eq!(err_pt.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(blob_count(&base), before_pt);
        let _ = in_b;

        // 3. Scheduler integration check: duplicate node.outputs fails node cleanly with zero extra CAS blobs!
        let sched_dir = base.join("sched");
        let run_id = RunId::from_bytes([0xE9; 16]);
        let bundle = RunArtifactBundle::create(&sched_dir, run_id).unwrap();
        let mut session = DataOpsSession::new(Arc::new(RunArtifactSink::new(bundle)));
        let run_dir = session.sink().bundle().run_dir().to_path_buf();
        write_once_cas_blob(&run_dir, &parse_cas_uri_digest(&uri_a).unwrap(), &tbl_a).unwrap();
        let (hex_sched_b, uri_sched_b, _) = write_cas_bytes(&run_dir, &tbl_b).unwrap();
        let _ = hex_sched_b;
        let ingest = test_node("ingest");
        session
            .register_source(
                "dataset://ns/a",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: uri_a,
                    content_type: CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
                    auth_mode: AuthModeMarker::None,
                    etag_or_version: None,
                },
                None,
                &ingest,
            )
            .unwrap();
        session
            .register_source(
                "dataset://ns/b",
                TrustClass::Trusted,
                SourceDescriptorV0 {
                    uri: uri_sched_b,
                    content_type: CONTENT_TYPE_COLUMNAR_JSON_V1.to_string(),
                    auth_mode: AuthModeMarker::None,
                    etag_or_version: None,
                },
                None,
                &ingest,
            )
            .unwrap();
        let blobs_before_sched = blob_count(&run_dir);
        let mut sched_bad = test_node("filter_rows");
        sched_bad.node_key = "node/dup_out".to_string();
        sched_bad.inputs = vec![
            AssetRefV1 {
                asset_key: "dataset://ns/a".to_string(),
                fingerprint: None,
            },
            AssetRefV1 {
                asset_key: "dataset://ns/b".to_string(),
                fingerprint: None,
            },
        ];
        sched_bad.outputs = vec![
            AssetRefV1 {
                asset_key: "dataset://ns/dup_out".to_string(),
                fingerprint: None,
            },
            AssetRefV1 {
                asset_key: "dataset://ns/dup_out".to_string(),
                fingerprint: None,
            },
        ];
        sched_bad
            .params
            .insert("column".to_string(), CanonValue::Str("k".to_string()));
        sched_bad
            .params
            .insert("op".to_string(), CanonValue::Str("ge".to_string()));
        sched_bad
            .params
            .insert("value".to_string(), CanonValue::Str("2".to_string()));
        let graph = GraphV1 {
            schema_version: 1,
            graph_id: Some("dup-out-sched".to_string()),
            nodes: vec![sched_bad],
            edges: vec![],
        };
        let r = execute_graph_sequential(
            &graph,
            &mut session,
            &runner,
            &PermissivePolicy,
            run_id,
            ctx.clock_nanos,
        )
        .unwrap();
        assert_eq!(r.failed_nodes, vec!["node/dup_out"]);
        assert_eq!(blob_count(&run_dir), blobs_before_sched);
        let _ = (hex_a, hex_b);
        drop(session);
        let _ = std::fs::remove_dir_all(&base);
    }
}
