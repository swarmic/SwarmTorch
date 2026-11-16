//! Executable JSON Schema Draft 2020-12 validator for SwarmTorch v1 artifacts
//! (`ADR-0016`, `ADR-0023` §3.3, `P2-20a`, `W10-02`).
//!
//! Evaluates the canonical `schemas/v1/*.schema.json` contracts directly against
//! JSON and NDJSON values using `serde_json::Value`, preserving exact 64-bit integer
//! (`i64`/`u64`) and finite float ordering without `> 2^53` coercion, Unicode scalar
//! `minLength`/`maxLength` counts, and linear-time size-bounded `regex` patterns.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::io;

use serde_json::Value;

use super::bundle::VerifiedBundleV1;
use super::io::{cmp_json_numbers, MAX_NDJSON_LINES, MAX_NDJSON_LINE_BYTES};

const MAX_SCHEMA_REGEX_LEN: usize = 4_096;
const REGEX_COMPILE_SIZE_LIMIT_BYTES: usize = 1 << 20;

/// Expected JSON Schema meta-schema URI for Draft 2020-12.
pub const JSON_SCHEMA_DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

/// Embedded canonical schema for `manifest.json`.
pub const MANIFEST_SCHEMA_JSON: &str = include_str!("../../schemas/v1/manifest.schema.json");
/// Embedded canonical schema for `run.json`.
pub const RUN_SCHEMA_JSON: &str = include_str!("../../schemas/v1/run.schema.json");
/// Embedded canonical schema for `graph.json`.
pub const GRAPH_SCHEMA_JSON: &str = include_str!("../../schemas/v1/graph.schema.json");
/// Embedded canonical schema for `datasets/registry.json`.
pub const REGISTRY_SCHEMA_JSON: &str = include_str!("../../schemas/v1/registry.schema.json");
/// Embedded canonical schema for `datasets/lineage.json`.
pub const LINEAGE_SCHEMA_JSON: &str = include_str!("../../schemas/v1/lineage.schema.json");
/// Embedded canonical schema for `spans.ndjson` lines.
pub const SPAN_SCHEMA_JSON: &str = include_str!("../../schemas/v1/span.schema.json");
/// Embedded canonical schema for `events.ndjson` lines.
pub const EVENT_SCHEMA_JSON: &str = include_str!("../../schemas/v1/event.schema.json");
/// Embedded canonical schema for `metrics.ndjson` lines.
pub const METRIC_SCHEMA_JSON: &str = include_str!("../../schemas/v1/metric.schema.json");
/// Embedded canonical schema for `datasets/materializations.ndjson` lines.
pub const MATERIALIZATION_SCHEMA_JSON: &str =
    include_str!("../../schemas/v1/materialization.schema.json");
/// Embedded canonical schema for `stcl-json-v1` columnar table payloads (`P2-20b`).
pub const COLUMNAR_TABLE_SCHEMA_JSON: &str =
    include_str!("../../schemas/v1/columnar_table.schema.json");

/// Error returned when a JSON/NDJSON artifact violates its Draft 2020-12 schema contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaValidationError {
    /// JSON pointer / path where the violation occurred.
    pub instance_path: String,
    /// Human-readable reason for the schema violation.
    pub message: String,
}

impl core::fmt::Display for SchemaValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "schema validation failed at '{}': {}",
            self.instance_path, self.message
        )
    }
}

impl std::error::Error for SchemaValidationError {}

impl From<SchemaValidationError> for io::Error {
    fn from(err: SchemaValidationError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, err.to_string())
    }
}

/// Validate that a schema document itself has a valid Draft 2020-12 header and `$id`.
pub fn parse_and_check_schema_meta(schema_str: &str) -> Result<Value, SchemaValidationError> {
    let root: Value = serde_json::from_str(schema_str).map_err(|e| SchemaValidationError {
        instance_path: "$".to_string(),
        message: format!("invalid schema JSON: {e}"),
    })?;
    let obj = root.as_object().ok_or_else(|| SchemaValidationError {
        instance_path: "$".to_string(),
        message: "schema root must be an object".to_string(),
    })?;
    let meta = obj
        .get("$schema")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaValidationError {
            instance_path: "$.$schema".to_string(),
            message: "missing $schema declaration".to_string(),
        })?;
    if meta != JSON_SCHEMA_DRAFT_2020_12 {
        return Err(SchemaValidationError {
            instance_path: "$.$schema".to_string(),
            message: format!("expected {JSON_SCHEMA_DRAFT_2020_12}, got {meta}"),
        });
    }
    let id = obj
        .get("$id")
        .and_then(Value::as_str)
        .ok_or_else(|| SchemaValidationError {
            instance_path: "$.$id".to_string(),
            message: "missing $id URI".to_string(),
        })?;
    if !id.starts_with("https://swarmtorch.dev/schemas/v1/") {
        return Err(SchemaValidationError {
            instance_path: "$.$id".to_string(),
            message: format!("unexpected $id prefix: {id}"),
        });
    }
    Ok(root)
}

/// Validate a JSON byte slice against a JSON Schema 2020-12 contract string.
pub fn validate_json_bytes_against_schema(
    json_bytes: &[u8],
    schema_str: &str,
) -> Result<(), SchemaValidationError> {
    let root_schema = parse_and_check_schema_meta(schema_str)?;
    let instance: Value =
        serde_json::from_slice(json_bytes).map_err(|e| SchemaValidationError {
            instance_path: "$".to_string(),
            message: format!("malformed JSON: {e}"),
        })?;
    let mut regex_cache = HashMap::new();
    validate_value_node(&instance, &root_schema, &root_schema, "$", &mut regex_cache)
}

/// Validate an NDJSON byte slice line-by-line against a JSON Schema 2020-12 line contract.
pub fn validate_ndjson_bytes_against_schema(
    ndjson_bytes: &[u8],
    schema_str: &str,
) -> Result<(), SchemaValidationError> {
    let root_schema = parse_and_check_schema_meta(schema_str)?;
    let mut regex_cache = HashMap::new();
    let mut line_count = 0usize;
    for (idx, raw_line) in ndjson_bytes.split(|&b| b == b'\n').enumerate() {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        if line.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        if line.len() > MAX_NDJSON_LINE_BYTES {
            return Err(SchemaValidationError {
                instance_path: format!("$[{idx}]"),
                message: format!(
                    "NDJSON line length {} exceeds maximum {}",
                    line.len(),
                    MAX_NDJSON_LINE_BYTES
                ),
            });
        }
        line_count += 1;
        if line_count > MAX_NDJSON_LINES {
            return Err(SchemaValidationError {
                instance_path: format!("$[{idx}]"),
                message: format!("NDJSON line count exceeds maximum {MAX_NDJSON_LINES}"),
            });
        }
        let instance: Value = serde_json::from_slice(line).map_err(|e| SchemaValidationError {
            instance_path: format!("$[{idx}]"),
            message: format!("malformed NDJSON line: {e}"),
        })?;
        validate_value_node(
            &instance,
            &root_schema,
            &root_schema,
            &format!("$[{idx}]"),
            &mut regex_cache,
        )?;
    }
    Ok(())
}

/// Validate all v1 JSON and NDJSON artifacts in a [`VerifiedBundleV1`] against their
/// canonical Draft 2020-12 schemas (`P2-20a`).
pub fn validate_verified_bundle_schemas(
    verified: &VerifiedBundleV1,
) -> Result<(), SchemaValidationError> {
    let manifest_bytes =
        serde_json::to_vec(verified.manifest()).map_err(|e| SchemaValidationError {
            instance_path: "manifest.json".to_string(),
            message: e.to_string(),
        })?;
    validate_json_bytes_against_schema(&manifest_bytes, MANIFEST_SCHEMA_JSON)?;

    let json_targets = [
        ("run.json", RUN_SCHEMA_JSON),
        ("graph.json", GRAPH_SCHEMA_JSON),
        ("datasets/registry.json", REGISTRY_SCHEMA_JSON),
        ("datasets/lineage.json", LINEAGE_SCHEMA_JSON),
    ];
    for (rel_path, schema) in json_targets {
        let bytes = verified
            .require_bytes(rel_path)
            .map_err(|e| SchemaValidationError {
                instance_path: rel_path.to_string(),
                message: e.to_string(),
            })?;
        validate_json_bytes_against_schema(bytes, schema)?;
    }

    let ndjson_targets = [
        ("spans.ndjson", SPAN_SCHEMA_JSON),
        ("events.ndjson", EVENT_SCHEMA_JSON),
        ("metrics.ndjson", METRIC_SCHEMA_JSON),
        (
            "datasets/materializations.ndjson",
            MATERIALIZATION_SCHEMA_JSON,
        ),
        (
            "datasets/registry_updates.ndjson",
            &dataset_entry_line_schema_json(),
        ),
        (
            "datasets/lineage_edges.ndjson",
            &lineage_edge_line_schema_json(),
        ),
    ];
    for (rel_path, schema) in ndjson_targets {
        if let Some(bytes) = verified.get_bytes(rel_path) {
            validate_ndjson_bytes_against_schema(bytes, schema)?;
        }
    }

    Ok(())
}

fn dataset_entry_line_schema_json() -> String {
    let root: Value = serde_json::from_str(REGISTRY_SCHEMA_JSON).expect("valid registry schema");
    let entry_def = root["$defs"]["DatasetEntryV1"].clone();
    let defs = root["$defs"].clone();
    let mut line_schema = entry_def;
    line_schema["$schema"] = Value::String(JSON_SCHEMA_DRAFT_2020_12.to_string());
    line_schema["$id"] =
        Value::String("https://swarmtorch.dev/schemas/v1/registry_update.schema.json".to_string());
    line_schema["$defs"] = defs;
    serde_json::to_string(&line_schema).expect("serializable schema")
}

fn lineage_edge_line_schema_json() -> String {
    let root: Value = serde_json::from_str(LINEAGE_SCHEMA_JSON).expect("valid lineage schema");
    let mut line_schema = root["$defs"]["LineageEdgeV1"].clone();
    line_schema["$schema"] = Value::String(JSON_SCHEMA_DRAFT_2020_12.to_string());
    line_schema["$id"] =
        Value::String("https://swarmtorch.dev/schemas/v1/lineage_edge.schema.json".to_string());
    serde_json::to_string(&line_schema).expect("serializable schema")
}

fn resolve_local_ref<'a>(
    ref_str: &str,
    root_schema: &'a Value,
    path: &str,
) -> Result<&'a Value, SchemaValidationError> {
    let def_name = ref_str
        .strip_prefix("#/$defs/")
        .ok_or_else(|| SchemaValidationError {
            instance_path: path.to_string(),
            message: format!("unsupported $ref URI: {ref_str}"),
        })?;
    root_schema
        .get("$defs")
        .and_then(|d| d.get(def_name))
        .ok_or_else(|| SchemaValidationError {
            instance_path: path.to_string(),
            message: format!("unresolved $ref definition: {ref_str}"),
        })
}

fn matches_type_name(instance: &Value, expected_type: &str) -> bool {
    match expected_type {
        "object" => instance.is_object(),
        "array" => instance.is_array(),
        "string" => instance.is_string(),
        "boolean" => instance.is_boolean(),
        "null" => instance.is_null(),
        "integer" => instance
            .as_number()
            .map(|n| n.is_i64() || n.is_u64())
            .unwrap_or(false),
        "number" => instance
            .as_number()
            .map(|n| n.as_f64().map(f64::is_finite).unwrap_or(false))
            .unwrap_or(false),
        _ => false,
    }
}

fn matches_pattern(
    s: &str,
    pattern: &str,
    path: &str,
    regex_cache: &mut HashMap<String, regex::Regex>,
) -> Result<bool, SchemaValidationError> {
    if pattern.len() > MAX_SCHEMA_REGEX_LEN {
        return Err(SchemaValidationError {
            instance_path: path.to_string(),
            message: format!(
                "schema regex pattern length {} exceeds maximum {MAX_SCHEMA_REGEX_LEN}",
                pattern.len()
            ),
        });
    }
    if let Some(re) = regex_cache.get(pattern) {
        return Ok(re.is_match(s));
    }
    let compiled = regex::RegexBuilder::new(pattern)
        .size_limit(REGEX_COMPILE_SIZE_LIMIT_BYTES)
        .dfa_size_limit(REGEX_COMPILE_SIZE_LIMIT_BYTES)
        .build()
        .map_err(|e| SchemaValidationError {
            instance_path: path.to_string(),
            message: format!("invalid schema regex pattern '{pattern}': {e}"),
        })?;
    let matched = compiled.is_match(s);
    regex_cache.insert(pattern.to_string(), compiled);
    Ok(matched)
}

fn validate_value_node(
    instance: &Value,
    schema: &Value,
    root_schema: &Value,
    path: &str,
    regex_cache: &mut HashMap<String, regex::Regex>,
) -> Result<(), SchemaValidationError> {
    let schema_obj = schema.as_object().ok_or_else(|| SchemaValidationError {
        instance_path: path.to_string(),
        message: "schema node must be a JSON object".to_string(),
    })?;

    if let Some(ref_val) = schema_obj.get("$ref").and_then(Value::as_str) {
        let target = resolve_local_ref(ref_val, root_schema, path)?;
        return validate_value_node(instance, target, root_schema, path, regex_cache);
    }

    if let Some(one_of) = schema_obj.get("oneOf").and_then(Value::as_array) {
        let mut matched_count = 0usize;
        let mut last_err = None;
        for candidate in one_of {
            match validate_value_node(instance, candidate, root_schema, path, regex_cache) {
                Ok(()) => matched_count += 1,
                Err(e) => last_err = Some(e),
            }
        }
        if matched_count != 1 {
            return Err(last_err.unwrap_or_else(|| SchemaValidationError {
                instance_path: path.to_string(),
                message: format!("expected exactly 1 matching oneOf schema, got {matched_count}"),
            }));
        }
    }

    if let Some(any_of) = schema_obj.get("anyOf").and_then(Value::as_array) {
        let mut matched = false;
        let mut last_err = None;
        for candidate in any_of {
            match validate_value_node(instance, candidate, root_schema, path, regex_cache) {
                Ok(()) => {
                    matched = true;
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        if !matched {
            return Err(last_err.unwrap_or_else(|| SchemaValidationError {
                instance_path: path.to_string(),
                message: "value did not match any schema in anyOf".to_string(),
            }));
        }
    }

    // Type check
    if let Some(type_val) = schema_obj.get("type") {
        let type_ok = match type_val {
            Value::String(t) => matches_type_name(instance, t),
            Value::Array(arr) => arr
                .iter()
                .filter_map(Value::as_str)
                .any(|t| matches_type_name(instance, t)),
            _ => false,
        };
        if !type_ok {
            return Err(SchemaValidationError {
                instance_path: path.to_string(),
                message: format!("value {instance} does not match expected type {type_val}"),
            });
        }
    }

    // Const check
    if let Some(const_val) = schema_obj.get("const") {
        if instance != const_val {
            return Err(SchemaValidationError {
                instance_path: path.to_string(),
                message: format!("expected const {const_val}, got {instance}"),
            });
        }
    }

    // Enum check
    if let Some(enum_arr) = schema_obj.get("enum").and_then(Value::as_array) {
        if !enum_arr.iter().any(|candidate| candidate == instance) {
            return Err(SchemaValidationError {
                instance_path: path.to_string(),
                message: format!("value {instance} is not in allowed enum set"),
            });
        }
    }

    // Numeric bounds (lossless i64 / u64 / finite f64 without > 2^53 precision coercion)
    if let Some(num) = instance.as_number() {
        if let Some(min_val) = schema_obj.get("minimum").and_then(Value::as_number) {
            let ord = cmp_json_numbers(num, min_val).ok_or_else(|| SchemaValidationError {
                instance_path: path.to_string(),
                message: format!("non-finite numeric comparison for {num} vs minimum {min_val}"),
            })?;
            if ord == Ordering::Less {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("number {num} < minimum {min_val}"),
                });
            }
        }
        if let Some(max_val) = schema_obj.get("maximum").and_then(Value::as_number) {
            let ord = cmp_json_numbers(num, max_val).ok_or_else(|| SchemaValidationError {
                instance_path: path.to_string(),
                message: format!("non-finite numeric comparison for {num} vs maximum {max_val}"),
            })?;
            if ord == Ordering::Greater {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("number {num} > maximum {max_val}"),
                });
            }
        }
        if let Some(ex_min) = schema_obj
            .get("exclusiveMinimum")
            .and_then(Value::as_number)
        {
            let ord = cmp_json_numbers(num, ex_min).ok_or_else(|| SchemaValidationError {
                instance_path: path.to_string(),
                message: format!(
                    "non-finite numeric comparison for {num} vs exclusiveMinimum {ex_min}"
                ),
            })?;
            if ord != Ordering::Greater {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("number {num} <= exclusiveMinimum {ex_min}"),
                });
            }
        }
        if let Some(ex_max) = schema_obj
            .get("exclusiveMaximum")
            .and_then(Value::as_number)
        {
            let ord = cmp_json_numbers(num, ex_max).ok_or_else(|| SchemaValidationError {
                instance_path: path.to_string(),
                message: format!(
                    "non-finite numeric comparison for {num} vs exclusiveMaximum {ex_max}"
                ),
            })?;
            if ord != Ordering::Less {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("number {num} >= exclusiveMaximum {ex_max}"),
                });
            }
        }
    }

    // String constraints (Draft 2020-12 §6.3.1-6.3.2 Unicode scalar count + linear-time regex)
    if let Some(s) = instance.as_str() {
        let char_len = s.chars().count() as u64;
        if let Some(min_len) = schema_obj.get("minLength").and_then(Value::as_u64) {
            if char_len < min_len {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("string char length {char_len} < minLength {min_len}"),
                });
            }
        }
        if let Some(max_len) = schema_obj.get("maxLength").and_then(Value::as_u64) {
            if char_len > max_len {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("string char length {char_len} > maxLength {max_len}"),
                });
            }
        }
        if let Some(pattern) = schema_obj.get("pattern").and_then(Value::as_str) {
            if !matches_pattern(s, pattern, path, regex_cache)? {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("string '{s}' does not match pattern {pattern}"),
                });
            }
        }
    }

    // Array constraints
    if let Some(arr) = instance.as_array() {
        if let Some(min_items) = schema_obj.get("minItems").and_then(Value::as_u64) {
            if (arr.len() as u64) < min_items {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("array length {} < minItems {min_items}", arr.len()),
                });
            }
        }
        if let Some(max_items) = schema_obj.get("maxItems").and_then(Value::as_u64) {
            if (arr.len() as u64) > max_items {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!("array length {} > maxItems {max_items}", arr.len()),
                });
            }
        }
        if let Some(item_schema) = schema_obj.get("items") {
            for (idx, item) in arr.iter().enumerate() {
                let child_path = format!("{path}[{idx}]");
                validate_value_node(item, item_schema, root_schema, &child_path, regex_cache)?;
            }
        }
    }

    // Object constraints
    if let Some(obj) = instance.as_object() {
        if let Some(max_props) = schema_obj.get("maxProperties").and_then(Value::as_u64) {
            if (obj.len() as u64) > max_props {
                return Err(SchemaValidationError {
                    instance_path: path.to_string(),
                    message: format!(
                        "object property count {} > maxProperties {max_props}",
                        obj.len()
                    ),
                });
            }
        }
        if let Some(req_arr) = schema_obj.get("required").and_then(Value::as_array) {
            for req in req_arr.iter().filter_map(Value::as_str) {
                if !obj.contains_key(req) {
                    return Err(SchemaValidationError {
                        instance_path: path.to_string(),
                        message: format!("missing required property '{req}'"),
                    });
                }
            }
        }

        let props = schema_obj.get("properties").and_then(Value::as_object);
        let additional = schema_obj.get("additionalProperties");

        for (k, v) in obj {
            let child_path = format!("{path}.{k}");
            if let Some(prop_schema) = props.and_then(|p| p.get(k)) {
                validate_value_node(v, prop_schema, root_schema, &child_path, regex_cache)?;
            } else if let Some(Value::Bool(false)) = additional {
                return Err(SchemaValidationError {
                    instance_path: child_path,
                    message: format!("unexpected additional property '{k}'"),
                });
            } else if let Some(add_schema @ Value::Object(_)) = additional {
                validate_value_node(v, add_schema, root_schema, &child_path, regex_cache)?;
            }
        }
    }

    Ok(())
}
