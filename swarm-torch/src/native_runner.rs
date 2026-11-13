//! Minimal native OpRunner (alpha.6, std-only).
//!
//! Implements three metadata-only ops:
//! - `passthrough`: forwards inputs unchanged
//! - `filter_rows`: filters rows (metadata-only; rows/bytes = None)
//! - `union`: forwards input metadata for union-style stages; materialization computes outputs
//!
//! All ops emit a deterministic span:
//! - `trace_id = run_id` (16 bytes → TraceId)
//! - `span_id = sha256(node_id_bytes || ts_nanos_be)[0..8]`
//!
//! **ADR-0018:** The runner boundary is separate from the scheduler.
//! Policy enforcement must happen BEFORE calling `run()`.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use swarm_torch_core::execution::{AssetInstanceV1, OpRunner};
use swarm_torch_core::observe::{AttrMap, RunEventEmitter, RunId, SpanId, SpanRecord, TraceId};
use swarm_torch_core::run_graph::{CanonParams, CanonValue, NodeV1};

use crate::artifacts::schema_check::{
    validate_json_bytes_against_schema, COLUMNAR_TABLE_SCHEMA_JSON,
};
use crate::artifacts::{
    hex_lower, parse_cas_uri_digest, read_bytes_bounded, write_once_cas_blob, OutputSpec,
    MAX_SINGLE_ARTIFACT_BYTES,
};
use crate::wasm_sandbox::{SandboxedExtensionPolicyV1, WasmSandboxRunnerV1};

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
        let body_bytes = serde_json::to_vec(&(&table.columns, &table.rows))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let digest = Sha256::digest(&body_bytes);
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&digest[..]);
        table.content_digest_sha256 = Some(hex_lower(&arr));
        Ok(table)
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
        if self.row_count != self.rows.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ColumnarTableV1 row_count does not match rows.len()",
            ));
        }
        for (col_idx, col) in self.columns.iter().enumerate() {
            if col.name.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("empty column name at index {col_idx}"),
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
        Ok(())
    }
}

/// Borrowed Wasm sandbox policy + module registry context passed to [`NativeOpRunner::run_in_session`].
pub type WasmIsolationCtx<'a> = (
    &'a SandboxedExtensionPolicyV1,
    &'a BTreeMap<[u8; 32], Vec<u8>>,
);

/// Output produced by [`NativeOpRunner::run_in_session`], preserving physical row/byte counts
/// and content-addressed CAS URIs alongside [`AssetInstanceV1`] descriptors (`W12-04`).
#[derive(Debug, Clone)]
pub struct PhysicalExecOutcomeV1 {
    /// Emitted output asset instances (including `cas://sha256/<hex>` URIs).
    pub instances: Vec<AssetInstanceV1>,
    /// Computed output specs carrying real physical row/byte measurements.
    pub output_specs: Vec<OutputSpec>,
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

/// Minimal native OpRunner (metadata-only).
///
/// Supports three op_types:
/// - `"passthrough"` — returns inputs as-is
/// - `"filter_rows"` — returns inputs with metadata indicating filter applied
/// - `"union"` — returns input metadata unchanged (output derivation happens at materialization)
pub struct NativeOpRunner;

impl NativeOpRunner {
    /// Run with explicit execution context.
    ///
    /// Emits a span for the operation. Returns output asset instances.
    pub fn run_with_context<E: RunEventEmitter<Error = io::Error>>(
        &self,
        ctx: &ExecutionContext,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        emitter: &E,
    ) -> io::Result<Vec<AssetInstanceV1>> {
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

    /// Execute a graph operator against physical CAS blobs in `run_dir` (`ColumnarTableV1` JSON /
    /// NDJSON) and/or fail-closed WebAssembly extensions (`W12-04`).
    ///
    /// Falls back cleanly to metadata-only `run_with_context` when inputs have no `cas://sha256/`
    /// physical blobs and the operator is a core metadata transform.
    pub fn run_in_session<E: RunEventEmitter<Error = io::Error>>(
        &self,
        ctx: &ExecutionContext,
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
        emitter: &E,
        wasm_isolation: Option<WasmIsolationCtx<'_>>,
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let start_nanos = (ctx.clock_nanos)();
        let node_id = node
            .node_id
            .unwrap_or_else(|| swarm_torch_core::run_graph::node_id_from_key(&node.node_key));
        let span_id = deterministic_span_id(node_id.as_bytes(), start_nanos);
        let trace_id = TraceId::from_bytes(*ctx.run_id.as_bytes());

        if node.op_type == "wasm_transform" {
            let (policy, registry) = wasm_isolation.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "wasm_transform requires SandboxedExtensionPolicyV1 isolation",
                )
            })?;
            let hex_digest = canon_param_str(&node.params, "module_sha256").ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "wasm_transform missing required param 'module_sha256'",
                )
            })?;
            let module_hash = decode_sha256_hex(hex_digest)?;
            let wasm_bytes = registry.get(&module_hash).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("wasm module {hex_digest} missing from sandbox registry"),
                )
            })?;
            let export_name = canon_param_str(&node.params, "export").unwrap_or("transform");

            let in_bytes = if let Some(first) = inputs.first() {
                if first
                    .uri
                    .as_deref()
                    .and_then(parse_cas_uri_digest)
                    .is_some()
                {
                    read_verified_cas_input(run_dir, first)?
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            let receipt = WasmSandboxRunnerV1::execute_byte_transform(
                policy,
                wasm_bytes,
                export_name,
                &in_bytes,
            )
            .map_err(|e| io::Error::new(io::ErrorKind::PermissionDenied, e.to_string()))?;

            let (out_hex, out_uri, out_bytes) = write_cas_bytes(run_dir, &receipt.output_bytes)?;
            let out_key = node
                .outputs
                .first()
                .map(|o| o.asset_key.clone())
                .unwrap_or_else(|| format!("asset://wasm/{out_hex}"));
            let instances = vec![AssetInstanceV1 {
                asset_key: out_key.clone(),
                fingerprint_v0: out_hex,
                uri: Some(out_uri),
            }];
            let output_specs = vec![OutputSpec {
                asset_key: out_key,
                schema: None,
                rows: None,
                bytes: Some(out_bytes),
            }];
            let end_nanos = (ctx.clock_nanos)();
            Self::emit_op_span(
                trace_id,
                span_id,
                node,
                inputs.len(),
                instances.len(),
                (start_nanos, end_nanos),
                emitter,
            )?;
            return Ok(PhysicalExecOutcomeV1 {
                instances,
                output_specs,
            });
        }

        let has_cas_inputs = !inputs.is_empty()
            && inputs
                .iter()
                .all(|i| i.uri.as_deref().and_then(parse_cas_uri_digest).is_some());

        if !has_cas_inputs {
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
            });
        }

        // Physical CAS execution (`passthrough`, `filter_rows`, `union`)
        let outcome = match node.op_type.as_str() {
            "passthrough" => Self::physical_passthrough(run_dir, node, inputs)?,
            "filter_rows" => Self::physical_filter_rows(run_dir, node, inputs)?,
            "union" => Self::physical_union(run_dir, node, inputs)?,
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

    fn physical_passthrough(
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let mut instances = Vec::with_capacity(inputs.len());
        let mut output_specs = Vec::with_capacity(inputs.len());
        for (idx, inp) in inputs.iter().enumerate() {
            let raw = read_verified_cas_input(run_dir, inp)?;
            let (hex, uri, byte_len) = write_cas_bytes(run_dir, &raw)?;
            let row_count = infer_row_count(&raw);
            let out_key = node
                .outputs
                .get(idx)
                .map(|o| o.asset_key.clone())
                .unwrap_or_else(|| inp.asset_key.clone());
            instances.push(AssetInstanceV1 {
                asset_key: out_key.clone(),
                fingerprint_v0: hex,
                uri: Some(uri),
            });
            output_specs.push(OutputSpec {
                asset_key: out_key,
                schema: None,
                rows: row_count,
                bytes: Some(byte_len),
            });
        }
        Ok(PhysicalExecOutcomeV1 {
            instances,
            output_specs,
        })
    }

    fn physical_filter_rows(
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
    ) -> io::Result<PhysicalExecOutcomeV1> {
        let col_name = canon_param_str(&node.params, "column");
        let op = canon_param_str(&node.params, "op").unwrap_or("eq");
        let target_val = canon_param_str(&node.params, "value");

        let mut instances = Vec::with_capacity(inputs.len());
        let mut output_specs = Vec::with_capacity(inputs.len());

        for (idx, inp) in inputs.iter().enumerate() {
            let raw = read_verified_cas_input(run_dir, inp)?;
            let (out_bytes, rows_kept) = if raw.first().copied() == Some(b'{') {
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
            } else {
                // Bounded NDJSON line filtering
                let text = std::str::from_utf8(&raw)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                let mut out_lines = Vec::new();
                for line in text.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let obj: Value = serde_json::from_str(trimmed)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                    let keep = match (col_name, target_val) {
                        (Some(col), Some(rhs)) => obj
                            .get(col)
                            .is_some_and(|cell| cell_matches_predicate(cell, op, rhs)),
                        _ => true,
                    };
                    if keep {
                        out_lines.push(trimmed.to_string());
                    }
                }
                let count = out_lines.len() as u64;
                let mut buf = out_lines.join("\n").into_bytes();
                if !buf.is_empty() {
                    buf.push(b'\n');
                }
                (buf, count)
            };

            let (hex, uri, byte_len) = write_cas_bytes(run_dir, &out_bytes)?;
            let out_key = node
                .outputs
                .get(idx)
                .map(|o| o.asset_key.clone())
                .unwrap_or_else(|| inp.asset_key.clone());
            instances.push(AssetInstanceV1 {
                asset_key: out_key.clone(),
                fingerprint_v0: hex,
                uri: Some(uri),
            });
            output_specs.push(OutputSpec {
                asset_key: out_key,
                schema: None,
                rows: Some(rows_kept),
                bytes: Some(byte_len),
            });
        }
        Ok(PhysicalExecOutcomeV1 {
            instances,
            output_specs,
        })
    }

    fn physical_union(
        run_dir: &Path,
        node: &NodeV1,
        inputs: &[AssetInstanceV1],
    ) -> io::Result<PhysicalExecOutcomeV1> {
        if inputs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "union requires at least one physical input",
            ));
        }

        let first_raw = read_verified_cas_input(run_dir, &inputs[0])?;
        let (out_bytes, total_rows) = if first_raw.first().copied() == Some(b'{') {
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
        } else {
            let mut merged_lines = Vec::new();
            for inp in inputs {
                let raw = read_verified_cas_input(run_dir, inp)?;
                let text = std::str::from_utf8(&raw)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
                for line in text.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        merged_lines.push(trimmed.to_string());
                    }
                }
            }
            let total_rows = merged_lines.len() as u64;
            let mut buf = merged_lines.join("\n").into_bytes();
            if !buf.is_empty() {
                buf.push(b'\n');
            }
            (buf, total_rows)
        };

        let (hex, uri, byte_len) = write_cas_bytes(run_dir, &out_bytes)?;
        let out_key = node
            .outputs
            .first()
            .map(|o| o.asset_key.clone())
            .unwrap_or_else(|| format!("asset://union/{hex}"));
        Ok(PhysicalExecOutcomeV1 {
            instances: vec![AssetInstanceV1 {
                asset_key: out_key.clone(),
                fingerprint_v0: hex,
                uri: Some(uri),
            }],
            output_specs: vec![OutputSpec {
                asset_key: out_key,
                schema: None,
                rows: Some(total_rows),
                bytes: Some(byte_len),
            }],
        })
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

fn decode_sha256_hex(s: &str) -> io::Result<[u8; 32]> {
    if s.len() != 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected 64-char sha256 hex, got {} chars", s.len()),
        ));
    }
    let b = s.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hex_val(b[i * 2])?;
        let lo = hex_val(b[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_val(c: u8) -> io::Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid hex character",
        )),
    }
}

fn infer_row_count(raw: &[u8]) -> Option<u64> {
    if raw.first().copied() == Some(b'{') {
        if let Ok(tbl) = ColumnarTableV1::from_json_bytes_strict(raw) {
            return Some(tbl.row_count);
        }
    }
    if let Ok(text) = std::str::from_utf8(raw) {
        let lines = text.lines().filter(|l| !l.trim().is_empty()).count() as u64;
        if lines > 0 {
            return Some(lines);
        }
    }
    None
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
    if let Some(lhs_f) = cell.as_f64() {
        if let Ok(rhs_f) = rhs_raw.parse::<f64>() {
            if !lhs_f.is_finite() || !rhs_f.is_finite() {
                return false;
            }
            return match op {
                "eq" => (lhs_f - rhs_f).abs() <= 1e-12,
                "ne" => (lhs_f - rhs_f).abs() > 1e-12,
                "gt" => lhs_f > rhs_f,
                "ge" => lhs_f >= rhs_f,
                "lt" => lhs_f < rhs_f,
                "le" => lhs_f <= rhs_f,
                _ => false,
            };
        }
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
        let runner = NativeOpRunner;
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
        let runner = NativeOpRunner;
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
        let runner = NativeOpRunner;
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
        let runner = NativeOpRunner;
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
        let runner = NativeOpRunner;
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

        let runner = NativeOpRunner;
        let ctx = ExecutionContext {
            run_id: RunId::from_bytes([7; 16]),
            clock_nanos: || 1_000_000,
        };
        let emitter = TestEmitter::new();

        let filter_out = runner
            .run_in_session(
                &ctx,
                &run_dir,
                &filter_node,
                std::slice::from_ref(&in1),
                &emitter,
                None,
            )
            .unwrap();
        assert_eq!(filter_out.output_specs[0].rows, Some(2));
        assert!(filter_out.output_specs[0].bytes.unwrap() > 0);
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
        let mut union_node = test_node("union");
        union_node.outputs = vec![AssetRefV1 {
            asset_key: "dataset://ns/merged".to_string(),
            fingerprint: None,
        }];
        let union_out = runner
            .run_in_session(
                &ctx,
                &run_dir,
                &union_node,
                &[filter_out.instances[0].clone(), in2],
                &emitter,
                None,
            )
            .unwrap();
        assert_eq!(union_out.output_specs[0].rows, Some(3));

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
        let err = runner
            .run_in_session(&ctx, &run_dir, &union_node, &[in1, in_bad], &emitter, None)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let _ = std::fs::remove_dir_all(&run_dir);
    }
}
