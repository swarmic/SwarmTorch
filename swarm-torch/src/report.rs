//! Standalone HTML report generator (artifact reader).
//!
//! This is intentionally offline-first:
//! - reads a run artifact bundle directory
//! - validates `manifest.json`
//! - generates a self-contained `report.html` without requiring a server/DB/UI framework

use std::cmp::Ordering;
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use swarm_torch_core::dataops::{
    DatasetLineageV1, DatasetRegistryV1, MaterializationRecordV1, TrustClass,
};
use swarm_torch_core::observe::{EventRecord, MetricRecord, SpanRecord};
use swarm_torch_core::run_graph::{ExecutionTrust, GraphV1, NodeId, NodeV1};

use crate::artifacts::RunArtifactBundle;

/// Report data loaded from a run artifact bundle.
#[derive(Debug, serde::Serialize)]
pub struct Report {
    #[serde(serialize_with = "serialize_path")]
    pub run_dir: PathBuf,
    pub graph: GraphV1,
    pub registry: DatasetRegistryV1,
    pub lineage: DatasetLineageV1,
    pub materializations: Vec<MaterializationRecordV1>,
    pub spans: Vec<SpanRecord>,
    pub events: Vec<EventRecord>,
    pub metrics: Vec<MetricRecord>,
}

fn serialize_path<S: serde::Serializer>(path: &PathBuf, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&path.display().to_string())
}

pub fn load_report(run_dir: impl AsRef<Path>) -> io::Result<Report> {
    let run_dir = run_dir.as_ref().to_path_buf();
    let bundle = RunArtifactBundle::open(&run_dir)?;

    // Enforce tamper-evidence by default.
    bundle.validate_manifest()?;

    let mut graph: GraphV1 = read_json(run_dir.join("graph.json"))?;
    graph = graph.normalize();

    let registry: DatasetRegistryV1 = read_json(run_dir.join("datasets").join("registry.json"))?;
    let lineage: DatasetLineageV1 = read_json(run_dir.join("datasets").join("lineage.json"))?;

    let spans: Vec<SpanRecord> = read_ndjson(run_dir.join("spans.ndjson"))?;
    let events: Vec<EventRecord> = read_ndjson(run_dir.join("events.ndjson"))?;
    let metrics: Vec<MetricRecord> = read_ndjson(run_dir.join("metrics.ndjson"))?;
    let materializations: Vec<MaterializationRecordV1> =
        read_ndjson(run_dir.join("datasets").join("materializations.ndjson"))?;

    Ok(Report {
        run_dir,
        graph,
        registry,
        lineage,
        materializations,
        spans,
        events,
        metrics,
    })
}

pub fn generate_report_html(
    run_dir: impl AsRef<Path>,
    out_path: impl AsRef<Path>,
) -> io::Result<()> {
    let report = load_report(run_dir)?;
    let html = render_html(&report);
    fs::write(out_path, html)
}

/// Generate report with optional JSON output.
///
/// If `json_out` is Some, writes pretty-printed JSON alongside HTML.
pub fn generate_report(
    run_dir: impl AsRef<Path>,
    html_out: impl AsRef<Path>,
    json_out: Option<impl AsRef<Path>>,
) -> io::Result<()> {
    let report = load_report(&run_dir)?;
    let html = render_html(&report);
    fs::write(&html_out, html)?;

    if let Some(json_path) = json_out {
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(json_path, json)?;
    }

    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<T> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_ndjson<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<Vec<T>> {
    let f = fs::File::open(path)?;
    let reader = io::BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v = serde_json::from_str::<T>(&line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid ndjson at line {}: {}", i + 1, e),
            )
        })?;
        out.push(v);
    }
    Ok(out)
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
        .replace('\'', "&#39;")
}

fn node_index_map(graph: &GraphV1) -> std::collections::HashMap<NodeId, usize> {
    let mut m = std::collections::HashMap::new();
    for (i, n) in graph.nodes.iter().enumerate() {
        if let Some(id) = n.node_id {
            m.insert(id, i);
        }
    }
    m
}

/// Derive whether a node should be marked unsafe.
///
/// A node is unsafe if:
/// - `execution_trust != Core`, OR
/// - any input asset_key is `Untrusted` in the registry, OR
/// - any input asset_key is **missing** from the registry (fail closed).
pub fn is_node_unsafe(node: &NodeV1, registry: &DatasetRegistryV1) -> bool {
    if node.execution_trust != ExecutionTrust::Core {
        return true;
    }
    for input in &node.inputs {
        match registry
            .datasets
            .iter()
            .find(|d| d.asset_key == input.asset_key)
        {
            Some(ds) if ds.trust == TrustClass::Untrusted => return true,
            None => return true, // missing input → fail closed
            _ => {}
        }
    }
    false
}

fn render_svg(graph: &GraphV1, registry: &DatasetRegistryV1) -> String {
    let width = 900;
    let node_w = 820;
    let node_h = 56;
    let x0 = 40;
    let y0 = 30;
    let y_step = 86;
    let height = y0 + (graph.nodes.len().max(1) * y_step) + 30;

    let idx = node_index_map(graph);

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\">"
    ));
    svg.push_str("<style>.n{font:14px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace}.s{stroke:#222;stroke-width:2;fill:#fff}.u{stroke:#b00020;stroke-width:3}.e{stroke:#666;stroke-width:2;fill:none;marker-end:url(#a)}</style>");
    svg.push_str("<defs><marker id=\"a\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#666\"/></marker></defs>");

    // Edges (if present).
    for e in &graph.edges {
        let Some(&from_i) = idx.get(&e.from_node_id) else {
            continue;
        };
        let Some(&to_i) = idx.get(&e.to_node_id) else {
            continue;
        };
        let x1 = x0 + node_w;
        let y1 = y0 + from_i * y_step + node_h / 2;
        let x2 = x0;
        let y2 = y0 + to_i * y_step + node_h / 2;
        svg.push_str(&format!(
            "<path class=\"e\" d=\"M {x1} {y1} C {x1} {y1} {x2} {y2} {x2} {y2}\"/>"
        ));
    }

    // Nodes — use derived is_node_unsafe instead of just n.unsafe_surface.
    for (i, n) in graph.nodes.iter().enumerate() {
        let x = x0;
        let y = y0 + i * y_step;
        let derived_unsafe = is_node_unsafe(n, registry);
        let cls = if derived_unsafe { "s u" } else { "s" };
        svg.push_str(&format!(
            "<rect x=\"{x}\" y=\"{y}\" rx=\"10\" ry=\"10\" width=\"{node_w}\" height=\"{node_h}\" class=\"{cls}\"/>"
        ));
        let title = format!(
            "{}  [{}:{}]{}",
            n.node_key,
            format!("{:?}", n.op_kind).to_lowercase(),
            n.op_type,
            if derived_unsafe { "  UNSAFE" } else { "" }
        );
        svg.push_str(&format!(
            "<text class=\"n\" x=\"{}\" y=\"{}\">{}</text>",
            x + 16,
            y + 34,
            escape_html(&title)
        ));
    }

    if graph.nodes.is_empty() {
        svg.push_str("<text class=\"n\" x=\"40\" y=\"60\">(graph.json has no nodes yet)</text>");
    }

    svg.push_str("</svg>");
    svg
}

#[derive(Debug, Clone)]
struct TimelineRow {
    ts: u64,
    kind: &'static str,
    name: String,
    detail: String,
}

fn render_timeline(report: &Report) -> String {
    let mut rows: Vec<TimelineRow> = Vec::new();
    let mut node_unsafe_by_id: std::collections::HashMap<NodeId, bool> =
        std::collections::HashMap::new();
    for node in &report.graph.nodes {
        if let Some(node_id) = node.node_id {
            node_unsafe_by_id.insert(node_id, is_node_unsafe(node, &report.registry));
        }
    }

    for e in &report.events {
        rows.push(TimelineRow {
            ts: e.ts_unix_nanos,
            kind: "event",
            name: e.name.clone(),
            detail: String::new(),
        });
    }
    for m in &report.metrics {
        rows.push(TimelineRow {
            ts: m.ts_unix_nanos,
            kind: "metric",
            name: m.name.clone(),
            detail: format!("value={} {}", m.value, m.unit.as_deref().unwrap_or("")),
        });
    }
    for s in &report.spans {
        rows.push(TimelineRow {
            ts: s.start_unix_nanos,
            kind: "span",
            name: s.name.clone(),
            detail: match s.end_unix_nanos {
                Some(end) if end >= s.start_unix_nanos => {
                    format!("duration_ms={}", (end - s.start_unix_nanos) / 1_000_000)
                }
                _ => "duration_ms=?".to_string(),
            },
        });
    }
    for m in &report.materializations {
        let derived_unsafe = node_unsafe_by_id.get(&m.node_id).copied().unwrap_or(true);
        rows.push(TimelineRow {
            ts: m.ts_unix_nanos,
            kind: "materialization",
            name: m.asset_key.clone(),
            detail: format!(
                "rows={} bytes={} cache_hit={} unsafe={} node_id={} node_def_hash={}",
                m.rows
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                m.bytes
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                m.cache_hit
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "?".to_string()),
                derived_unsafe,
                m.node_id,
                &m.node_def_hash
            ),
        });
    }

    rows.sort_by(|a, b| match a.ts.cmp(&b.ts) {
        Ordering::Equal => a.kind.cmp(b.kind),
        o => o,
    });

    let mut out = String::new();
    out.push_str("<table><thead><tr><th>ts_unix_nanos</th><th>kind</th><th>name</th><th>detail</th></tr></thead><tbody>");
    for r in rows {
        out.push_str(&format!(
            "<tr><td class=\"mono\">{}</td><td>{}</td><td class=\"mono\">{}</td><td class=\"mono\">{}</td></tr>",
            r.ts,
            escape_html(r.kind),
            escape_html(&r.name),
            escape_html(&r.detail)
        ));
    }
    out.push_str("</tbody></table>");
    out
}

fn render_html(report: &Report) -> String {
    // Derive unsafe nodes using is_node_unsafe (registry-aware)
    let mut unsafe_nodes = Vec::new();
    for n in &report.graph.nodes {
        if is_node_unsafe(n, &report.registry) {
            unsafe_nodes.push(n.node_key.clone());
        }
    }

    let mut unsafe_datasets = Vec::new();
    for d in &report.registry.datasets {
        if matches!(d.trust, swarm_torch_core::dataops::TrustClass::Untrusted) {
            unsafe_datasets.push(d.asset_key.clone());
        }
    }

    let mut unsafe_materializations = Vec::new();
    for m in &report.materializations {
        if m.unsafe_surface {
            unsafe_materializations.push(m.asset_key.clone());
        }
    }

    let mut html = String::new();
    html.push_str("<!doctype html><html><head><meta charset=\"utf-8\"/>");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>");
    html.push_str("<title>SwarmTorch Run Report</title>");
    html.push_str("<style>body{font:15px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111}h1,h2{margin:18px 0 10px}code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,monospace;font-size:13px}table{border-collapse:collapse;width:100%;margin:8px 0 16px}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#fafafa;text-align:left}section{margin:18px 0 22px}.warn{border:2px solid #b00020;padding:10px;border-radius:10px;background:#fff5f5}.ok{border:2px solid #2e7d32;padding:10px;border-radius:10px;background:#f5fff7}</style>");
    html.push_str("</head><body>");
    html.push_str("<h1>SwarmTorch Run Report</h1>");

    html.push_str(&format!(
        "<p><strong>Run dir:</strong> <code>{}</code></p>",
        escape_html(&report.run_dir.display().to_string())
    ));

    if unsafe_nodes.is_empty() && unsafe_datasets.is_empty() && unsafe_materializations.is_empty() {
        html.push_str("<div class=\"ok\"><strong>Unsafe surfaces:</strong> none detected in the current artifacts.</div>");
    } else {
        html.push_str("<div class=\"warn\"><strong>Unsafe surfaces detected.</strong><ul>");
        for n in unsafe_nodes {
            html.push_str(&format!("<li>node: <code>{}</code></li>", escape_html(&n)));
        }
        for d in unsafe_datasets {
            html.push_str(&format!(
                "<li>dataset source untrusted: <code>{}</code></li>",
                escape_html(&d)
            ));
        }
        for m in unsafe_materializations {
            html.push_str(&format!(
                "<li>unsafe materialization: <code>{}</code></li>",
                escape_html(&m)
            ));
        }
        html.push_str("</ul></div>");
    }

    html.push_str("<section><h2>Run Graph</h2>");
    html.push_str(&render_svg(&report.graph, &report.registry));
    html.push_str("</section>");

    html.push_str("<section><h2>Timeline</h2>");
    html.push_str(&render_timeline(report));
    html.push_str("</section>");

    html.push_str("<section><h2>Dataset Registry</h2>");
    html.push_str("<table><thead><tr><th>asset_key</th><th>fingerprint_v0</th><th>trust</th><th>source</th></tr></thead><tbody>");
    for d in &report.registry.datasets {
        html.push_str(&format!(
            "<tr><td class=\"mono\">{}</td><td class=\"mono\">{}</td><td>{:?}</td><td class=\"mono\">{}</td></tr>",
            escape_html(&d.asset_key),
            escape_html(&d.fingerprint_v0),
            d.trust,
            escape_html(
                &d.source
                    .as_ref()
                    .map(|s| s.uri.clone())
                    .unwrap_or_else(|| "".to_string())
            ),
        ));
    }
    html.push_str("</tbody></table></section>");

    html.push_str("<section><h2>Lineage</h2>");
    html.push_str("<table><thead><tr><th>input_fingerprint</th><th>output_fingerprint</th><th>node_id</th><th>op_kind</th></tr></thead><tbody>");
    for e in &report.lineage.edges {
        html.push_str(&format!(
            "<tr><td class=\"mono\">{}</td><td class=\"mono\">{}</td><td class=\"mono\">{}</td><td>{:?}</td></tr>",
            escape_html(&e.input_fingerprint_v0),
            escape_html(&e.output_fingerprint_v0),
            escape_html(&e.node_id.to_string()),
            e.op_kind
        ));
    }
    html.push_str("</tbody></table></section>");

    html.push_str("</body></html>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::dataops::{DatasetEntryV1, DatasetLineageV1, TrustClass};
    use swarm_torch_core::observe::TraceId;
    use swarm_torch_core::run_graph::{AssetRefV1, CanonParams, ExecutionTrust, NodeV1, OpKind};

    fn make_node(key: &str, trust: ExecutionTrust, inputs: &[&str]) -> NodeV1 {
        NodeV1 {
            node_key: key.to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "test".to_string(),
            inputs: inputs
                .iter()
                .map(|k| AssetRefV1 {
                    asset_key: k.to_string(),
                    fingerprint: None,
                })
                .collect(),
            outputs: vec![],
            params: CanonParams::new(),
            code_ref: Some("test@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: trust,
            node_def_hash: None,
        }
    }

    fn make_entry(asset_key: &str, trust: TrustClass) -> DatasetEntryV1 {
        DatasetEntryV1 {
            asset_key: asset_key.to_string(),
            fingerprint_v0: "a".repeat(64),
            source_fingerprint_v0: "b".repeat(64),
            schema_hash_v0: "c".repeat(64),
            recipe_hash_v0: "d".repeat(64),
            trust,
            source: None,
            schema: None,
            license_flags: vec![],
            pii_tags: vec![],
        }
    }

    #[test]
    fn report_marks_node_unsafe_on_untrusted_input() {
        let node = make_node(
            "transform/clean",
            ExecutionTrust::Core,
            &["dataset://ns/raw"],
        );
        let registry = DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![make_entry("dataset://ns/raw", TrustClass::Untrusted)],
        };

        assert!(
            is_node_unsafe(&node, &registry),
            "node with untrusted input should be marked unsafe"
        );
    }

    #[test]
    fn report_marks_node_unsafe_on_missing_input_registry_entry() {
        // Node references an input that doesn't exist in the registry at all
        let node = make_node(
            "transform/clean",
            ExecutionTrust::Core,
            &["dataset://ns/missing"],
        );
        let registry = DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![], // empty registry
        };

        assert!(
            is_node_unsafe(&node, &registry),
            "node with missing input in registry should be marked unsafe (fail closed)"
        );
    }

    #[test]
    fn report_node_safe_with_core_trust_and_trusted_inputs() {
        let node = make_node(
            "transform/clean",
            ExecutionTrust::Core,
            &["dataset://ns/raw"],
        );
        let registry = DatasetRegistryV1 {
            schema_version: 1,
            datasets: vec![make_entry("dataset://ns/raw", TrustClass::Trusted)],
        };

        assert!(
            !is_node_unsafe(&node, &registry),
            "Core trust node with all trusted inputs should not be unsafe"
        );
    }

    #[test]
    fn timeline_materialization_includes_node_id_and_node_def_hash() {
        let node_id = TraceId::from_bytes([1u8; 16]);
        let mut node = make_node(
            "transform/clean",
            ExecutionTrust::Core,
            &["dataset://ns/missing"],
        );
        node.node_id = Some(node_id);

        let mat = MaterializationRecordV1 {
            schema_version: 1,
            ts_unix_nanos: 1000,
            asset_key: "dataset://ns/out".to_string(),
            fingerprint_v0: "x".repeat(64),
            node_id,
            node_def_hash: "h".repeat(64),
            rows: Some(100),
            bytes: Some(1000),
            cache_hit: Some(false),
            duration_ms: Some(50),
            quality_flags: None,
            unsafe_surface: false, // intentionally false: timeline should derive from node+registry.
        };

        let report = Report {
            run_dir: PathBuf::from("/tmp/test"),
            graph: GraphV1 {
                schema_version: 1,
                graph_id: None,
                nodes: vec![node],
                edges: vec![],
            },
            registry: DatasetRegistryV1 {
                schema_version: 1,
                datasets: vec![], // missing input => derived unsafe=true
            },
            lineage: DatasetLineageV1 {
                schema_version: 1,
                edges: vec![],
            },
            materializations: vec![mat],
            spans: vec![],
            events: vec![],
            metrics: vec![],
        };
        let timeline_html = render_timeline(&report);

        assert!(
            timeline_html.contains("node_id="),
            "timeline detail should include node_id: {timeline_html}"
        );
        assert!(
            timeline_html.contains("node_def_hash="),
            "timeline detail should include node_def_hash: {timeline_html}"
        );
        assert!(
            timeline_html.contains(&"h".repeat(64)),
            "timeline detail should include full node_def_hash value: {timeline_html}"
        );
        assert!(
            timeline_html.contains("unsafe=true"),
            "timeline unsafe should be derived from node+registry, not materialization flag: {timeline_html}"
        );
    }
}
