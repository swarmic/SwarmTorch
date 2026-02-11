//! Minimal native OpRunner (alpha.6, std-only).
//!
//! Implements three metadata-only ops:
//! - `passthrough`: forwards inputs unchanged
//! - `filter_rows`: filters rows (metadata-only; rows/bytes = None)
//! - `union`: unions multiple inputs (metadata-only; rows/bytes = None)
//!
//! All ops emit a deterministic span:
//! - `trace_id = run_id` (16 bytes → TraceId)
//! - `span_id = sha256(node_id_bytes || ts_nanos_be)[0..8]`
//!
//! **ADR-0018:** The runner boundary is separate from the scheduler.
//! Policy enforcement must happen BEFORE calling `run()`.

use std::collections::BTreeMap;
use std::io;

use sha2::{Digest, Sha256};

use swarm_torch_core::execution::{AssetInstanceV1, OpRunner};
use swarm_torch_core::observe::{AttrMap, RunEventEmitter, RunId, SpanId, SpanRecord, TraceId};
use swarm_torch_core::run_graph::NodeV1;

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
/// - `"union"` — returns a single output combining all input asset_keys
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

        // Emit span
        let mut attrs: AttrMap = BTreeMap::new();
        attrs.insert(
            "op_type".to_string(),
            swarm_torch_core::observe::AttrValue::Str(node.op_type.clone()),
        );
        attrs.insert(
            "node_key".to_string(),
            swarm_torch_core::observe::AttrValue::Str(node.node_key.clone()),
        );
        attrs.insert(
            "input_count".to_string(),
            swarm_torch_core::observe::AttrValue::I64(inputs.len() as i64),
        );
        attrs.insert(
            "output_count".to_string(),
            swarm_torch_core::observe::AttrValue::I64(outputs.len() as i64),
        );

        let span = SpanRecord {
            schema_version: 1,
            trace_id,
            span_id,
            parent_span_id: None,
            name: format!("op/{}", node.op_type),
            start_unix_nanos: start_nanos,
            end_unix_nanos: Some(end_nanos),
            attrs,
        };
        emitter.emit_span(&span)?;

        Ok(outputs)
    }

    // ── Op implementations ──────────────────────────────────────────

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

    /// Union: combines all inputs into a single merged asset list.
    /// This is metadata-only; actual merging would happen in a real runner.
    fn op_union(inputs: &[AssetInstanceV1], _node: &NodeV1) -> Vec<AssetInstanceV1> {
        // For metadata-only: return all inputs (the orchestrator uses
        // materialize_node_outputs to create the actual output fingerprint).
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
        // Use a zero-filled RunId and system clock as fallback.
        let ctx = ExecutionContext {
            run_id: RunId::from_bytes([0u8; 16]),
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
}
