use std::cmp::Ordering;
use std::fs;

use swarm_torch_core::run_graph::{GraphV1, NodeId};

use super::load::load_report;
use super::model::{
    build_registry_trust_index, format_transform_names, format_unsafe_reasons,
    is_node_unsafe_with_index, Report,
};

pub fn generate_report_html(
    run_dir: impl AsRef<std::path::Path>,
    out_path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    let report = load_report(run_dir)?;
    let html = render_html(&report);
    fs::write(out_path, html)
}

/// Generate report with optional JSON output.
///
/// If `json_out` is Some, writes pretty-printed JSON alongside HTML.
pub fn generate_report(
    run_dir: impl AsRef<std::path::Path>,
    html_out: impl AsRef<std::path::Path>,
    json_out: Option<impl AsRef<std::path::Path>>,
) -> std::io::Result<()> {
    let report = load_report(&run_dir)?;
    let html = render_html(&report);
    fs::write(&html_out, html)?;

    if let Some(json_path) = json_out {
        let json = serde_json::to_string_pretty(&report).map_err(std::io::Error::other)?;
        fs::write(json_path, json)?;
    }

    Ok(())
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
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

fn render_svg(graph: &GraphV1, registry: &swarm_torch_core::dataops::DatasetRegistryV1) -> String {
    let width = 900;
    let node_w = 820;
    let node_h = 56;
    let x0 = 40;
    let y0 = 30;
    let y_step = 86;
    let height = y0 + (graph.nodes.len().max(1) * y_step) + 30;

    let idx = node_index_map(graph);
    let trust_index = build_registry_trust_index(registry);

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
        let derived_unsafe = is_node_unsafe_with_index(n, &trust_index);
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

pub(crate) fn render_timeline(report: &Report) -> String {
    let mut rows: Vec<TimelineRow> = Vec::new();
    let trust_index = build_registry_trust_index(&report.registry);
    let mut node_unsafe_by_id: std::collections::HashMap<NodeId, bool> =
        std::collections::HashMap::new();
    for node in &report.graph.nodes {
        if let Some(node_id) = node.node_id {
            node_unsafe_by_id.insert(node_id, is_node_unsafe_with_index(node, &trust_index));
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
        let unsafe_reasons = format_unsafe_reasons(&m.unsafe_reasons);
        let transforms = format_transform_names(&m.applied_transforms);
        rows.push(TimelineRow {
            ts: m.ts_unix_nanos,
            kind: "materialization",
            name: m.asset_key.clone(),
            detail: format!(
                "rows={} bytes={} cache_hit={} unsafe={} unsafe_reasons={} transforms={} node_id={} node_def_hash={}",
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
                unsafe_reasons,
                transforms,
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

pub(crate) fn render_html(report: &Report) -> String {
    let trust_index = build_registry_trust_index(&report.registry);
    // Derive unsafe nodes using is_node_unsafe (registry-aware)
    let mut unsafe_nodes = Vec::new();
    for n in &report.graph.nodes {
        if is_node_unsafe_with_index(n, &trust_index) {
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
            unsafe_materializations.push((
                m.asset_key.clone(),
                format_unsafe_reasons(&m.unsafe_reasons),
            ));
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
        for (asset_key, reasons) in unsafe_materializations {
            html.push_str(&format!(
                "<li>unsafe materialization: <code>{}</code> reasons=<code>{}</code></li>",
                escape_html(&asset_key),
                escape_html(&reasons)
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
